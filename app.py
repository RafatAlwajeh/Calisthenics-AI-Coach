# -*- coding: utf-8 -*-
"""
Gym Enterprise AI â€” foundation v4.1
Pose-based exercise analysis with Streamlit + Ultralytics YOLOv8-Pose.

Important scope:
- The current implementation analyzes elbow flexion/extension (e.g. a curl-like movement).
- YOLOv8-Pose does not provide 21 hand landmarks. Hand/finger analysis needs a separate,
  validated hand-landmark model; it is intentionally not faked here.
- st.camera_input captures a still image, not a continuous webcam stream.
"""

from __future__ import annotations

import os
import time
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ------------------------------- App setup -------------------------------

st.set_page_config(
    page_title="Gym Enterprise AI v4.1",
    page_icon="ðŸ’ª",
    layout="wide",
)

MODEL_NAME = os.getenv("POSE_MODEL", "yolov8n-pose.pt")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    html, body, [class*="css"] {font-family:'Cairo',Tahoma,sans-serif;}
    h1,h2,h3,p,label,.stMarkdown {direction:rtl;text-align:right;}
    section[data-testid="stSidebar"] {direction:rtl;}
    section[data-testid="stSidebar"] * {text-align:right;}
    .hero {padding:18px 22px;border-radius:16px;color:white;
           background:linear-gradient(120deg,#172554,#0f172a);
           border:1px solid #334155;margin-bottom:16px;direction:rtl;text-align:right;}
    div.stButton > button {width:100%;font-weight:700;border-radius:10px;min-height:2.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------- Model -----------------------------------

@st.cache_resource(show_spinner="Ø¬Ø§Ø±Ù ØªØ­Ù…ÙŠÙ„ Ù†Ù…ÙˆØ°Ø¬ ØªÙ‚Ø¯ÙŠØ± Ø§Ù„ÙˆØ¶Ø¹ÙŠØ©...")
def load_pose_model(model_name: str) -> YOLO:
    # First run may download the weights; deployment needs outbound access or a bundled model.
    return YOLO(model_name)


# -------------------------- Geometry and pose ----------------------------

# COCO keypoint indices for YOLOv8-Pose:
# left shoulder/elbow/wrist = 5/7/9; right = 6/8/10.
LEFT_ARM = (5, 7, 9)
RIGHT_ARM = (6, 8, 10)


def joint_confidence(kp_conf: Optional[np.ndarray], indices: Tuple[int, ...]) -> float:
    if kp_conf is None:
        return 0.0
    vals = np.asarray(kp_conf, dtype=float)[list(indices)]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) == len(indices) else 0.0


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """Return the angle ABC in degrees; None for degenerate/invalid geometry."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    if not all(np.isfinite(p).all() for p in (a, b, c)):
        return None
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return None
    cosine = float(np.dot(ba, bc) / (na * nc))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


@dataclass
class PersonPose:
    track_id: int
    angle: Optional[float]
    confidence: float
    side: str
    keypoints: np.ndarray


def extract_persons(
    result,
    min_joint_conf: float,
    require_track_ids: bool = True,
) -> List[PersonPose]:
    """Extract reliable arm poses.

    Video mode requires persistent tracker IDs. Still-image mode may use a
    frame-local index because identity persistence is not meaningful there.
    """
    poses: List[PersonPose] = []
    if result.keypoints is None or result.boxes is None:
        return poses

    xy = result.keypoints.xy
    conf_tensor = result.keypoints.conf
    if xy is None:
        return poses
    kps_all = xy.cpu().numpy()
    conf_all = conf_tensor.cpu().numpy() if conf_tensor is not None else None

    ids = None
    if result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()

    for i, kps in enumerate(kps_all):
        if ids is not None and i < len(ids):
            person_id = int(ids[i])
        elif require_track_ids:
            # In video mode, do not pretend array order is a persistent identity.
            continue
        else:
            # A still image has no persistent identity; this ID is frame-local only.
            person_id = i
        confs = conf_all[i] if conf_all is not None and i < len(conf_all) else None

        candidates = []
        for side, indices in (("left", LEFT_ARM), ("right", RIGHT_ARM)):
            conf = joint_confidence(confs, indices)
            if conf < min_joint_conf:
                continue
            shoulder, elbow, wrist = (kps[j] for j in indices)
            angle = calculate_angle(shoulder, elbow, wrist)
            if angle is not None:
                candidates.append((conf, angle, side))

        if not candidates:
            continue
        # Use the more reliable visible arm for this frame.
        conf, angle, side = max(candidates, key=lambda x: x[0])
        poses.append(PersonPose(person_id, angle, conf, side, kps))
    return poses


# -------------------------- Repetition engine ----------------------------

@dataclass
class RepState:
    stage: str = "up"
    start_time: Optional[float] = None
    min_angle: float = 180.0
    conf_sum: float = 0.0
    conf_count: int = 0


@dataclass
class UserState:
    rep: RepState = field(default_factory=RepState)


class RepQualityEngine:
    def __init__(
        self,
        depth_angle: float = 75,
        top_angle: float = 155,
        min_duration: float = 0.45,
        max_duration: float = 8.0,
        min_confidence: float = 0.55,
    ):
        self.depth_angle = depth_angle
        self.top_angle = top_angle
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_confidence = min_confidence
        self.users: Dict[int, UserState] = {}
        self.stats = {"performed": 0, "accepted": 0, "failed": 0, "review": 0}

    def update(self, person: PersonPose, timestamp: float) -> Optional[Tuple[str, str]]:
        if person.angle is None:
            return None
        state = self.users.setdefault(person.track_id, UserState())
        rep = state.rep
        angle = person.angle

        # Start a repetition only after a full extension/top position.
        if rep.stage == "up" and angle <= self.depth_angle:
            rep.stage = "down"
            rep.start_time = timestamp
            rep.min_angle = angle
            rep.conf_sum = person.confidence
            rep.conf_count = 1
            return None

        if rep.stage == "down":
            rep.min_angle = min(rep.min_angle, angle)
            rep.conf_sum += person.confidence
            rep.conf_count += 1

            if angle >= self.top_angle:
                duration = (
                    timestamp - rep.start_time
                    if rep.start_time is not None
                    else float("inf")
                )
                avg_conf = rep.conf_sum / max(1, rep.conf_count)
                depth_ok = rep.min_angle <= self.depth_angle
                tempo_ok = self.min_duration <= duration <= self.max_duration
                confidence_ok = avg_conf >= self.min_confidence

                if depth_ok and tempo_ok and confidence_ok:
                    verdict, message = "accepted", "ØªÙƒØ±Ø§Ø± Ù…Ù‚Ø¨ÙˆÙ„"
                elif not confidence_ok:
                    verdict, message = "review", "Ø¬ÙˆØ¯Ø© Ø§Ù„ØªØªØ¨Ø¹ ØºÙŠØ± ÙƒØ§ÙÙŠØ©"
                elif not depth_ok:
                    verdict, message = "failed", "Ù„Ù… ÙŠØªØ­Ù‚Ù‚ Ø§Ù„Ø¹Ù…Ù‚ Ø§Ù„Ù…Ø·Ù„ÙˆØ¨"
                elif not tempo_ok:
                    verdict, message = "review", "Ø²Ù…Ù† Ø§Ù„ØªÙƒØ±Ø§Ø± Ø®Ø§Ø±Ø¬ Ø§Ù„Ù†Ø·Ø§Ù‚"
                else:
                    verdict, message = "review", "ÙŠØ­ØªØ§Ø¬ Ø¥Ù„Ù‰ Ù…Ø±Ø§Ø¬Ø¹Ø©"

                self.stats["performed"] += 1
                self.stats[verdict] += 1
                state.rep = RepState()
                return verdict, message

        return None


# ---------------------------- Drawing ------------------------------------

def draw_overlay(frame: np.ndarray, poses: List[PersonPose], engine: RepQualityEngine,
                 elapsed: float, target: int, bot_reps: int, challenge_text: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw tracked arm landmarks and angle for each detected person.
    for pose in poses:
        indices = LEFT_ARM if pose.side == "left" else RIGHT_ARM
        points = [tuple(np.round(pose.keypoints[j]).astype(int)) for j in indices]
        for p in points:
            cv2.circle(out, p, 6, (0, 220, 255), -1)
        cv2.line(out, points[0], points[1], (0, 220, 255), 3)
        cv2.line(out, points[1], points[2], (0, 220, 255), 3)
        x, y = points[1]
        cv2.putText(out, f"ID {pose.track_id}: {pose.angle:.0f} deg",
                    (max(8, x - 60), max(25, y - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    accepted = engine.stats["accepted"]
    failed = engine.stats["failed"]
    review = engine.stats["review"]
    panel_h = min(128, h)
    cv2.rectangle(out, (0, 0), (w, panel_h), (18, 28, 45), -1)
    cv2.putText(out, f"YOU {accepted}/{target}", (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (70, 220, 100), 2, cv2.LINE_AA)
    cv2.putText(out, f"BOT {bot_reps}", (max(18, w - 180), 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"TIME {elapsed:.1f}s", (w // 2 - 65, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Accepted {accepted} | Failed {failed} | Review {review}",
                (18, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(out, challenge_text[:48], (18, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    # User progress bar.
    bar_x, bar_y, bar_w, bar_h = 18, min(h - 18, 120), max(20, w - 36), 10
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (65, 65, 65), -1)
    ratio = min(1.0, accepted / max(1, target))
    fill = int(bar_w * ratio)
    if fill:
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                      (50, 210, 90), -1)
    return out


def bot_count_at(elapsed: float, bot_interval: float) -> int:
    return max(0, int(elapsed // max(0.5, bot_interval)))


# ------------------------------ Sidebar ----------------------------------

st.sidebar.title("ðŸ‹ï¸ Ù…Ø±ÙƒØ² Ø§Ù„Ù‚ÙŠØ§Ø¯Ø©")
mode = st.sidebar.radio("ÙˆØ¶Ø¹ Ø§Ù„ØªØ´ØºÙŠÙ„", ["ØªØ­Ù„ÙŠÙ„ ÙÙŠØ¯ÙŠÙˆ", "ØªØ­Ù„ÙŠÙ„ ØµÙˆØ±Ø© Ø¨Ø§Ù„ÙƒØ§Ù…ÙŠØ±Ø§"])
st.sidebar.markdown("---")
target_reps = st.sidebar.slider("Ø§Ù„ØªÙƒØ±Ø§Ø±Ø§Øª Ø§Ù„Ù…Ø³ØªÙ‡Ø¯ÙØ©", 1, 50, 10)
depth_angle = st.sidebar.slider("Ø²Ø§ÙˆÙŠØ© Ø§Ù„Ø¹Ù…Ù‚ (Ø¯Ø±Ø¬Ø©)", 45, 110, 75)
top_angle = st.sidebar.slider("Ø²Ø§ÙˆÙŠØ© Ø§Ù„Ø¹ÙˆØ¯Ø© Ù„Ù„Ø£Ø¹Ù„Ù‰ (Ø¯Ø±Ø¬Ø©)", 140, 175, 155)
min_conf = st.sidebar.slider("Ø§Ù„Ø­Ø¯ Ø§Ù„Ø£Ø¯Ù†Ù‰ Ù„Ø«Ù‚Ø© Ø§Ù„Ù…ÙØ§ØµÙ„", 0.30, 0.90, 0.55, 0.05)
bot_interval = st.sidebar.slider("Ø²Ù…Ù† ØªÙƒØ±Ø§Ø± Ø§Ù„Ø¨ÙˆØª (Ø«Ø§Ù†ÙŠØ©)", 1.0, 10.0, 3.5, 0.5)
challenge_text = st.sidebar.text_input("Ø¹Ù†ÙˆØ§Ù† Ø§Ù„ØªØ­Ø¯ÙŠ", "Gym Enterprise AI Challenge")

st.markdown(
    """
    <div class="hero">
      <h2>ðŸ’ª Gym Enterprise AI</h2>
      <p>ØªØ­Ù„ÙŠÙ„ ÙˆØ¶Ø¹ÙŠØ© Ø§Ù„Ø¬Ø³Ù… ÙˆØ¬ÙˆØ¯Ø© Ø§Ù„ØªÙƒØ±Ø§Ø± Ø¨Ø§Ø³ØªØ®Ø¯Ø§Ù… YOLOv8-Pose.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------- Still camera mode ----------------------------

if mode == "ØªØ­Ù„ÙŠÙ„ ØµÙˆØ±Ø© Ø¨Ø§Ù„ÙƒØ§Ù…ÙŠØ±Ø§":
    st.info("Ù‡Ø°Ø§ Ø§Ù„ÙˆØ¶Ø¹ ÙŠÙ„ØªÙ‚Ø· ØµÙˆØ±Ø© Ø«Ø§Ø¨ØªØ© ÙˆÙŠØ­Ù„Ù„Ù‡Ø§Ø› Ù„Ø§ ÙŠÙ…Ø«Ù„ Ø¨Ø« ÙƒØ§Ù…ÙŠØ±Ø§ Ù…Ø³ØªÙ…Ø±Ø§Ù‹.")
    shot = st.camera_input("Ø§Ù„ØªÙ‚Ø· ØµÙˆØ±Ø© Ø¨ÙˆØ¶Ø¹ÙŠØ© Ø§Ù„ØªÙ…Ø±ÙŠÙ†")
    if shot is not None:
        raw = np.frombuffer(shot.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        try:
            model = load_pose_model(MODEL_NAME)
            result = model.predict(frame, verbose=False)[0]
            poses = extract_persons(result, min_conf, require_track_ids=False)
            annotated = result.plot()
            if poses:
                p = poses[0]
                cv2.putText(annotated, f"Elbow angle: {p.angle:.1f} deg ({p.side})",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2, cv2.LINE_AA)
                st.metric("Ø²Ø§ÙˆÙŠØ© Ø§Ù„ÙƒÙˆØ¹", f"{p.angle:.1f}Â°")
                st.caption(f"ØªØ­Ù„ÙŠÙ„ ØµÙˆØ±Ø© Ø«Ø§Ø¨ØªØ© (Ù„ÙŠØ³ ØªØªØ¨Ø¹Ø§Ù‹ Ù…Ø³ØªÙ…Ø±Ø§Ù‹)Ø› Ø«Ù‚Ø© Ø§Ù„Ù…ÙØ§ØµÙ„: {p.confidence:.2f}")
            else:
                st.warning("Ù„Ù… ØªÙÙƒØªØ´Ù Ø°Ø±Ø§Ø¹ Ù…ÙˆØ«ÙˆÙ‚Ø©. Ø£Ø¸Ù‡Ø± Ø§Ù„ÙƒØªÙ ÙˆØ§Ù„Ù…Ø±ÙÙ‚ ÙˆØ§Ù„Ø±Ø³Øº Ø¨ÙˆØ¶ÙˆØ­.")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        except Exception as exc:
            st.error(f"ØªØ¹Ø°Ø± ØªØ­Ù„ÙŠÙ„ Ø§Ù„ØµÙˆØ±Ø©: {exc}")


# ---------------------------- Video mode ---------------------------------

else:
    uploaded = st.file_uploader("Ø§Ø±ÙØ¹ ÙÙŠØ¯ÙŠÙˆ ØªÙ…Ø±ÙŠÙ†", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None:
        st.video(uploaded)
        if st.button("ðŸš€ Ø¨Ø¯Ø¡ ØªØ­Ù„ÙŠÙ„ Ø§Ù„ÙÙŠØ¯ÙŠÙˆ", type="primary"):
            engine = RepQualityEngine(
                depth_angle=depth_angle,
                top_angle=top_angle,
                min_confidence=min_conf,
            )
            progress = st.progress(0.0)
            status = st.empty()
            preview = st.empty()
            metrics = st.empty()

            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
            input_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            input_path, output_path = input_tmp.name, output_tmp.name
            input_tmp.write(uploaded.getvalue())
            input_tmp.close()
            output_tmp.close()

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if np.isfinite(fps) and fps > 0 else 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width <= 0 or height <= 0:
                cap.release()
                os.unlink(input_path)
                st.error("ØªØ¹Ø°Ø± Ù‚Ø±Ø§Ø¡Ø© Ø£Ø¨Ø¹Ø§Ø¯ Ø§Ù„ÙÙŠØ¯ÙŠÙˆ Ø£Ùˆ Ø£Ù† Ø§Ù„Ù…Ù„Ù ØºÙŠØ± ØµØ§Ù„Ø­.")
                st.stop()

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                cap.release()
                os.unlink(input_path)
                st.error("ØªØ¹Ø°Ø± Ø¥Ù†Ø´Ø§Ø¡ Ù…Ù„Ù Ø§Ù„ÙÙŠØ¯ÙŠÙˆ Ø§Ù„Ù†Ø§ØªØ¬. ØªØ­Ù‚Ù‚ Ù…Ù† Ø¯Ø¹Ù… ØªØ±Ù…ÙŠØ² MP4 ÙÙŠ Ø§Ù„Ø¨ÙŠØ¦Ø©.")
                st.stop()

            model = load_pose_model(MODEL_NAME)
            frame_index = 0
            last_message = "Ø§Ø¨Ø¯Ø£ Ø§Ù„ØªÙ…Ø±ÙŠÙ†"
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    timestamp = frame_index / fps  # Use media time, not wall-clock processing time.
                    result = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                        conf=0.25,
                    )[0]
                    poses = extract_persons(result, min_conf)

                    for pose in poses:
                        verdict = engine.update(pose, timestamp)
                        if verdict:
                            last_message = verdict[1]

                    bot_reps = bot_count_at(timestamp, bot_interval)
                    annotated = result.plot(boxes=True, labels=False)
                    annotated = draw_overlay(
                        annotated, poses, engine, timestamp, target_reps,
                        bot_reps, challenge_text,
                    )
                    writer.write(annotated)

                    if frame_index % max(1, int(fps // 4)) == 0:
                        preview.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            use_container_width=True,
                        )
                        stats = engine.stats
                        metrics.metric(
                            "Ø§Ù„ØªÙƒØ±Ø§Ø±Ø§Øª Ø§Ù„Ù…Ù‚Ø¨ÙˆÙ„Ø©",
                            f"{stats['accepted']} / {target_reps}",
                            delta=f"Ø§Ù„Ø¨ÙˆØª: {bot_reps}",
                        )
                    frame_index += 1
                    if total_frames > 0:
                        progress.progress(min(1.0, frame_index / total_frames))

                status.success("Ø§ÙƒØªÙ…Ù„ ØªØ­Ù„ÙŠÙ„ Ø§Ù„ÙÙŠØ¯ÙŠÙˆ.")
            except Exception as exc:
                status.error(f"Ø­Ø¯Ø« Ø®Ø·Ø£ Ø£Ø«Ù†Ø§Ø¡ Ø§Ù„Ù…Ø¹Ø§Ù„Ø¬Ø©: {exc}")
                st.exception(exc)
            finally:
                cap.release()
                writer.release()
                try:
                    os.unlink(input_path)
                except OSError:
                    pass

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                st.subheader("ðŸ“Š Ø§Ù„ØªÙ‚Ø±ÙŠØ± Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠ")
                cols = st.columns(4)
                for col, label, key in zip(
                    cols,
                    ["Ø§Ù„Ù…Ø­Ø§ÙˆÙ„Ø§Øª", "Ø§Ù„Ù…Ù‚Ø¨ÙˆÙ„Ø©", "Ø§Ù„Ù…Ø±ÙÙˆØ¶Ø©", "Ù‚ÙŠØ¯ Ø§Ù„Ù…Ø±Ø§Ø¬Ø¹Ø©"],
                    ["performed", "accepted", "failed", "review"],
                ):
                    col.metric(label, engine.stats[key])
                st.caption(f"Ø¢Ø®Ø± Ù…Ù„Ø§Ø­Ø¸Ø©: {last_message}")
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    "ðŸ“¥ ØªÙ†Ø²ÙŠÙ„ Ø§Ù„ÙÙŠØ¯ÙŠÙˆ Ø§Ù„Ù…Ø­Ù„Ù„",
                    data=video_bytes,
                    file_name="gym_enterprise_analyzed.mp4",
                    mime="video/mp4",
                )
                try:
                    os.unlink(output_path)
                except OSError:
                    pass

st.markdown("---")
st.caption(
    "Ù…Ù„Ø§Ø­Ø¸Ø© Ù‡Ù†Ø¯Ø³ÙŠØ©: Ø²ÙˆØ§ÙŠØ§ Ø§Ù„ÙƒÙˆØ¹ ÙˆØ­Ø¯Ù‡Ø§ Ù„Ø§ ØªØ«Ø¨Øª ØµØ­Ø© Ø§Ù„ØªÙ…Ø±ÙŠÙ† ÙƒØ§Ù…Ù„Ø§Ù‹. "
    "ÙŠØ¬Ø¨ Ù…Ø¹Ø§ÙŠØ±Ø© Ø§Ù„Ø¹ØªØ¨Ø§Øª Ø­Ø³Ø¨ Ù†ÙˆØ¹ Ø§Ù„Ø­Ø±ÙƒØ© ÙˆØ²Ø§ÙˆÙŠØ© Ø§Ù„ØªØµÙˆÙŠØ±ØŒ ÙˆØ§Ù„ØªØ­Ù‚Ù‚ Ø¹Ù„Ù‰ Ù…Ø¬Ù…ÙˆØ¹Ø© ÙÙŠØ¯ÙŠÙˆÙ‡Ø§Øª Ù…ÙˆØ³ÙˆÙ…Ø©."
)


import cv2

def draw_counter(frame, count, pos=(20, 40)):
    """
    Draw surfer counter on a frame.
    """
    out = frame.copy()
    cv2.putText(
        out,
        f"Surfers: {count}",
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),  # yellow
        3,
        cv2.LINE_AA,
    )
    return out

def draw_active_trackers(frame, active_tracks, pos=(0, 40)):
    """
    Draw surfer counter on a frame.
    """
    out = frame.copy()
    cv2.putText(
        out,
        f"IDs: {active_tracks}",
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),  # yellow
        1,
        cv2.LINE_AA,
    )
    return out

import cv2
def annotate_frame(
    frame,
    det,
    active_tracks,
    count,
    color=(0, 255, 0),
    thickness=2,
):
    annotated = frame.copy()

    # Draw count
    cv2.putText(
        annotated,
        f"Surfers: {count}",
        (400, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # Nothing to draw
    if det is None or det.boxes_xyxy is None or len(det.boxes_xyxy) == 0:
        return annotated

    for i, (x1, y1, x2, y2) in enumerate(det.boxes_xyxy):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        conf = float(det.scores[i]) if det.scores is not None else None

        # -----------------------------
        # Ultralytics (tracking exists)
        # -----------------------------
        if det.track_ids is not None:
            tid = int(det.track_ids[i])

            # Optional: only draw active tracks (same behavior as before)
            if tid not in active_tracks:
                continue

            label = f"ID {tid}"
            if conf is not None:
                label += f" ({conf:.2f})"

        # -----------------------------
        # SAHI (no tracking)
        # -----------------------------
        else:
            label = f"{conf:.2f}" if conf is not None else ""

        # Draw box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        # Draw label
        if label:
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return annotated
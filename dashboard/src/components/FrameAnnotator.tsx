import { useRef, useState, type PointerEvent } from "react";
import type { Box, Frame } from "../types";

type Point = { x: number; y: number };

type Props = {
  frame: Frame | null;
  boxes: Box[];
  selectedClass: string;
  onBoxesChange: (boxes: Box[]) => void;
};

const BOX_COLORS = ["#72f1b8", "#ffcf66", "#ff7a90", "#70b7ff", "#c995ff"];

export function FrameAnnotator({ frame, boxes, selectedClass, onBoxesChange }: Props) {
  const stageRef = useRef<HTMLDivElement>(null);
  const [start, setStart] = useState<Point | null>(null);
  const [current, setCurrent] = useState<Point | null>(null);

  const pointFromEvent = (event: PointerEvent): Point => {
    const bounds = stageRef.current?.getBoundingClientRect();
    if (!bounds) return { x: 0, y: 0 };
    return {
      x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
      y: Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height))
    };
  };

  const handlePointerDown = (event: PointerEvent) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = pointFromEvent(event);
    setStart(point);
    setCurrent(point);
  };

  const handlePointerMove = (event: PointerEvent) => {
    if (start) setCurrent(pointFromEvent(event));
  };

  const handlePointerUp = (event: PointerEvent) => {
    if (!start) return;
    const end = pointFromEvent(event);
    const width = Math.abs(end.x - start.x);
    const height = Math.abs(end.y - start.y);
    if (width > 0.01 && height > 0.01) {
      onBoxesChange([
        ...boxes,
        {
          class_name: selectedClass,
          x_center: Math.min(start.x, end.x) + width / 2,
          y_center: Math.min(start.y, end.y) + height / 2,
          width,
          height
        }
      ]);
    }
    setStart(null);
    setCurrent(null);
  };

  if (!frame) {
    return <div className="frame-empty">Select an episode and extract frames.</div>;
  }

  const draft = start && current ? toRectangle(start, current) : null;
  return (
    <div
      aria-label="Draw bounding boxes on the selected frame"
      className="frame-stage"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      ref={stageRef}
      role="application"
    >
      <img alt={`Episode frame ${frame.frame_index}`} draggable={false} src={frame.image_url} />
      <svg aria-hidden="true" className="box-layer" viewBox="0 0 100 100" preserveAspectRatio="none">
        {boxes.map((box, index) => (
          <g key={`${box.class_name}-${index}`}>
            <rect
              className={box.confidence == null ? "manual-box" : "prediction-box"}
              fill="transparent"
              height={box.height * 100}
              stroke={BOX_COLORS[index % BOX_COLORS.length]}
              vectorEffect="non-scaling-stroke"
              width={box.width * 100}
              x={(box.x_center - box.width / 2) * 100}
              y={(box.y_center - box.height / 2) * 100}
            />
          </g>
        ))}
        {draft ? (
          <rect
            className="draft-box"
            fill="transparent"
            height={draft.height * 100}
            vectorEffect="non-scaling-stroke"
            width={draft.width * 100}
            x={draft.x * 100}
            y={draft.y * 100}
          />
        ) : null}
      </svg>
    </div>
  );
}

function toRectangle(start: Point, end: Point) {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y)
  };
}

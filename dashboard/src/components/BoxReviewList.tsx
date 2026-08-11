import type { Dispatch, SetStateAction } from "react";
import type { Box } from "../types";

type Props = {
  boxes: Box[];
  classNames: string[];
  onChange: Dispatch<SetStateAction<Box[]>>;
};

const COORDINATES: Array<{ key: "x_center" | "y_center" | "width" | "height"; label: string }> = [
  { key: "x_center", label: "X" },
  { key: "y_center", label: "Y" },
  { key: "width", label: "W" },
  { key: "height", label: "H" }
];

export function BoxReviewList({ boxes, classNames, onChange }: Props) {
  const updateBox = (index: number, change: Partial<Box>) => {
    onChange((currentBoxes) =>
      currentBoxes.map((box, boxIndex) =>
        boxIndex === index ? { ...box, ...change } : box
      )
    );
  };

  const deleteBox = (index: number) => {
    onChange((currentBoxes) =>
      currentBoxes.filter((_, boxIndex) => boxIndex !== index)
    );
  };

  return (
    <div className="box-list">
      {boxes.map((box, index) => (
        <div
          className={`box-review ${box.suggested ? "suggested" : "manual"} ${box.accepted ? "accepted" : ""}`}
          key={`${box.class_name}-${index}`}
        >
          <div className="box-review-heading">
            <select
              aria-label={`Class for box ${index + 1}`}
              onChange={(event) => updateBox(index, { class_name: event.target.value })}
              value={box.class_name}
            >
              {classNames.map((className) => (
                <option key={className}>{className}</option>
              ))}
            </select>
            {box.confidence != null ? <small>{Math.round(box.confidence * 100)}%</small> : null}
            <button
              aria-label={`Delete ${box.class_name} box`}
              className="delete-box"
              onClick={() => deleteBox(index)}
              type="button"
            >
              ×
            </button>
          </div>
          <div className="coordinate-grid">
            {COORDINATES.map(({ key, label }) => (
              <label key={key}>
                {label}
                <input
                  aria-label={`${label} for box ${index + 1}`}
                  max={1}
                  min={key === "width" || key === "height" ? 0.01 : 0}
                  onChange={(event) => updateBox(index, { [key]: Number(event.target.value) })}
                  step={0.01}
                  type="number"
                  value={box[key].toFixed(3)}
                />
              </label>
            ))}
          </div>
          {box.suggested ? (
            <button
              aria-pressed={box.accepted === true}
              className="accept-box"
              onClick={() => updateBox(index, { accepted: box.accepted !== true })}
              type="button"
            >
              {box.accepted ? "Box approved" : "Approve box"}
            </button>
          ) : (
            <span className="manual-label">Manual box</span>
          )}
        </div>
      ))}
    </div>
  );
}

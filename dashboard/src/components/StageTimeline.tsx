import type { TaskStage } from "../types";

type Props = {
  stages: TaskStage[];
  currentStage?: string;
};

export function StageTimeline({ stages, currentStage }: Props) {
  return (
    <ol className="stage-timeline" aria-label="Approved task stages">
      {stages.map((stage) => (
        <li
          className={stage.name.toLowerCase() === currentStage?.toLowerCase() ? "current" : ""}
          key={`${stage.name}-${stage.start_time_seconds}`}
        >
          <span>{stage.name}</span>
          <small>{stage.start_time_seconds.toFixed(1)}–{stage.end_time_seconds.toFixed(1)}s</small>
        </li>
      ))}
    </ol>
  );
}

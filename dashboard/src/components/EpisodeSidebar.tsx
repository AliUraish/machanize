import type { Episode } from "../types";

type Props = {
  episodes: Episode[];
  selectedId?: string;
  onSelect: (episode: Episode) => void;
};

export function EpisodeSidebar({ episodes, selectedId, onSelect }: Props) {
  return (
    <aside className="episode-sidebar" aria-label="Recorded episodes">
      <div className="panel-heading">
        <span>Episodes</span>
        <span className="count-pill">{episodes.length}</span>
      </div>
      <div className="episode-list">
        {episodes.length === 0 ? (
          <p className="empty-copy">Copy a Pi recording into data/episodes to begin.</p>
        ) : (
          episodes.map((episode, index) => (
            <button
              className={`episode-card ${selectedId === episode.episode_id ? "selected" : ""}`}
              key={episode.episode_id}
              onClick={() => onSelect(episode)}
              type="button"
            >
              <span className="episode-number">EP {String(index + 1).padStart(2, "0")}</span>
              <strong>{episode.task || episode.project_name}</strong>
              <span>{episode.outcome.toUpperCase()}</span>
              <small>{episode.frame_count ?? "—"} recorded frames</small>
            </button>
          ))
        )}
      </div>
    </aside>
  );
}

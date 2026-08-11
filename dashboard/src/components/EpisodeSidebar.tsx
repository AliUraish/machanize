import type { Episode } from "../types";

type Props = {
  episodes: Episode[];
  activeId?: string;
  selectedIds: string[];
  onOpen: (episode: Episode) => void;
  onToggle: (episodeId: string) => void;
  onToggleAll: () => void;
};

export function EpisodeSidebar({
  episodes,
  activeId,
  selectedIds,
  onOpen,
  onToggle,
  onToggleAll
}: Props) {
  const selected = new Set(selectedIds);
  const allSelected = episodes.length > 0 && selectedIds.length === episodes.length;
  return (
    <aside className="episode-sidebar" aria-label="Recorded episodes">
      <div className="panel-heading">
        <span>Episodes</span>
        <button className="select-all" onClick={onToggleAll} type="button">
          {allSelected ? "Clear" : "Select all"}
        </button>
      </div>
      <p className="selection-count">{selectedIds.length} of {episodes.length} selected</p>
      <div className="episode-list">
        {episodes.length === 0 ? (
          <p className="empty-copy">Copy a Pi recording into data/episodes to begin.</p>
        ) : (
          episodes.map((episode, index) => (
            <div
              className={`episode-card ${activeId === episode.episode_id ? "active" : ""} ${selected.has(episode.episode_id) ? "selected" : ""}`}
              key={episode.episode_id}
            >
              <input
                aria-label={`Select episode ${index + 1}`}
                checked={selected.has(episode.episode_id)}
                onChange={() => onToggle(episode.episode_id)}
                type="checkbox"
              />
              <button className="episode-open" onClick={() => onOpen(episode)} type="button">
                <span className="episode-number">EP {String(index + 1).padStart(2, "0")}</span>
                <strong>{episode.task || episode.project_name}</strong>
                <span>{episode.outcome.toUpperCase()}</span>
                <small>{episode.frame_count ?? "—"} recorded frames</small>
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}

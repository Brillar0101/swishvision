'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Film, Users, Eye, Plus, X } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { cn, formatFileSize } from '@/lib/utils';
import { jobsApi, TeamConfig, ProcessingConfig, VisualizationConfig } from '@/lib/api';
import { COLOR_PRESETS } from '@/lib/types';

// Header component
function Header() {
  return (
    <header className="flex items-center justify-between px-8 py-4 border-b border-baseline-100 bg-white">
      <div className="flex items-center gap-2">
        <span className="text-2xl font-bold text-blacktop">
          SWISH<span className="text-court">/</span>VISION
        </span>
      </div>
      <nav className="flex items-center gap-6">
        <a href="/videos" className="text-sm font-medium text-baseline-600 hover:text-blacktop transition-colors">
          My Videos
        </a>
        <button className="btn-secondary text-sm py-2 px-4">
          Sign In
        </button>
      </nav>
    </header>
  );
}

// Team configuration component
interface TeamConfigFormProps {
  label: string;
  team: TeamConfig;
  onChange: (team: TeamConfig) => void;
}

function TeamConfigForm({ label, team, onChange }: TeamConfigFormProps) {
  const [showRoster, setShowRoster] = useState(false);
  const [newJersey, setNewJersey] = useState('');
  const [newName, setNewName] = useState('');

  const addPlayer = () => {
    if (newJersey && newName) {
      onChange({
        ...team,
        roster: { ...team.roster, [newJersey]: newName },
      });
      setNewJersey('');
      setNewName('');
    }
  };

  const removePlayer = (jersey: string) => {
    const newRoster = { ...team.roster };
    delete newRoster[jersey];
    onChange({ ...team, roster: newRoster });
  };

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-baseline-600">{label}</label>

      {/* Team Name */}
      <input
        type="text"
        placeholder="Team name"
        value={team.name}
        onChange={(e) => onChange({ ...team, name: e.target.value })}
        className="input"
      />

      {/* Color Picker */}
      <div className="flex flex-wrap gap-2">
        {COLOR_PRESETS.map((color) => (
          <button
            key={color.hex}
            onClick={() => onChange({ ...team, color: color.hex })}
            className={cn(
              'w-8 h-8 rounded-full border-2 transition-all',
              team.color === color.hex ? 'border-blacktop' : 'border-transparent hover:border-baseline-300'
            )}
            style={{ backgroundColor: color.hex }}
            title={color.name}
          />
        ))}
      </div>

      {/* Roster Toggle */}
      <button
        onClick={() => setShowRoster(!showRoster)}
        className="flex items-center gap-2 text-sm text-court font-medium hover:underline"
      >
        <Plus size={14} />
        {showRoster ? 'Hide roster' : 'Add roster (optional)'}
      </button>

      {/* Roster Editor */}
      {showRoster && (
        <div className="space-y-2 p-4 bg-baseline-50 border border-baseline-200 rounded-lg">
          {Object.entries(team.roster).map(([jersey, name]) => (
            <div key={jersey} className="flex items-center gap-2">
              <span className="w-12 text-sm font-medium">#{jersey}</span>
              <span className="flex-1 text-sm">{name}</span>
              <button
                onClick={() => removePlayer(jersey)}
                className="p-1 text-baseline-400 hover:text-red-500"
              >
                <X size={14} />
              </button>
            </div>
          ))}
          <div className="flex items-center gap-2 pt-2 border-t border-baseline-200">
            <input
              type="text"
              placeholder="#"
              value={newJersey}
              onChange={(e) => setNewJersey(e.target.value)}
              className="w-16 input text-sm py-2"
            />
            <input
              type="text"
              placeholder="Player name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              className="flex-1 input text-sm py-2"
              onKeyDown={(e) => e.key === 'Enter' && addPlayer()}
            />
            <button onClick={addPlayer} className="btn-secondary text-sm py-2 px-3">
              Add
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Main upload page
export default function UploadPage() {
  const router = useRouter();
  const [video, setVideo] = useState<File | null>(null);
  const [teamA, setTeamA] = useState<TeamConfig>({ name: '', color: '#DC2626', roster: {} });
  const [teamB, setTeamB] = useState<TeamConfig>({ name: '', color: '#2563EB', roster: {} });
  const [durationMode, setDurationMode] = useState<'full' | 'custom'>('custom');
  const [maxSeconds, setMaxSeconds] = useState(60);
  const [options, setOptions] = useState({
    enableSegmentation: true,
    enableJerseyDetection: true,
    enableTacticalView: true,
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles[0]) {
      setVideo(acceptedFiles[0]);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/mp4': ['.mp4'],
      'video/quicktime': ['.mov'],
      'video/x-msvideo': ['.avi'],
    },
    maxFiles: 1,
    maxSize: 500 * 1024 * 1024, // 500MB
  });

  const handleSubmit = async () => {
    if (!video) {
      setError('Please select a video file');
      return;
    }
    if (!teamA.name || !teamB.name) {
      setError('Please enter team names');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const processing: ProcessingConfig = {
        max_duration_seconds: durationMode === 'custom' ? maxSeconds : undefined,
        enable_segmentation: options.enableSegmentation,
        enable_jersey_detection: options.enableJerseyDetection,
        enable_tactical_view: options.enableTacticalView,
      };

      const visualization: VisualizationConfig = {
        mask_display: options.enableSegmentation ? 'all' : 'none',
        show_player_names: options.enableJerseyDetection,
        show_jersey_numbers: options.enableJerseyDetection,
        tactical_view_position: options.enableTacticalView ? 'bottom-right' : 'none',
      };

      const result = await jobsApi.create(video, teamA, teamB, processing, visualization);
      router.push(`/videos/${result.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload video');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-sideline">
      <Header />

      <main className="w-full px-8 lg:px-16 py-16">
        <h1 className="text-4xl font-bold text-blacktop text-center mb-2">
          Analyze Your Game
        </h1>
        <p className="text-baseline-500 text-center mb-10">
          Upload a basketball video and get AI-powered analysis
        </p>

        {/* Video Upload */}
        <div
          {...getRootProps()}
          className={cn(
            'border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all',
            isDragActive
              ? 'border-court bg-court-50'
              : 'border-baseline-300 hover:border-court hover:bg-court-50/50',
            video && 'border-court bg-court-50'
          )}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-4">
            {video ? (
              <>
                <Film size={48} className="text-court" />
                <div>
                  <p className="font-medium text-blacktop">{video.name}</p>
                  <p className="text-sm text-baseline-500">{formatFileSize(video.size)}</p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setVideo(null);
                  }}
                  className="text-sm text-court hover:underline"
                >
                  Remove
                </button>
              </>
            ) : (
              <>
                <Upload size={48} className="text-baseline-400" />
                <div>
                  <p className="font-medium text-blacktop">
                    {isDragActive ? 'Drop your video here' : 'Drop your game footage here'}
                  </p>
                  <p className="text-sm text-baseline-500">or click to browse</p>
                </div>
                <p className="text-xs text-baseline-400">MP4, MOV up to 500MB</p>
              </>
            )}
          </div>
        </div>

        {/* Teams Section */}
        <div className="mt-8 space-y-6">
          <div className="section-label">Teams</div>
          <div className="grid grid-cols-2 gap-6">
            <TeamConfigForm label="Home Team" team={teamA} onChange={setTeamA} />
            <TeamConfigForm label="Away Team" team={teamB} onChange={setTeamB} />
          </div>
        </div>

        {/* Duration Section */}
        <div className="mt-8 space-y-4">
          <div className="section-label">Duration</div>
          <div className="space-y-3">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="radio"
                checked={durationMode === 'full'}
                onChange={() => setDurationMode('full')}
                className="w-4 h-4 text-court accent-court"
              />
              <span className="text-sm">Entire video</span>
            </label>
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="radio"
                checked={durationMode === 'custom'}
                onChange={() => setDurationMode('custom')}
                className="w-4 h-4 text-court accent-court"
              />
              <span className="text-sm">First</span>
              <input
                type="number"
                value={maxSeconds}
                onChange={(e) => setMaxSeconds(Number(e.target.value))}
                disabled={durationMode !== 'custom'}
                className="w-20 input text-sm py-1.5 px-3"
                min={10}
                max={600}
              />
              <span className="text-sm">seconds</span>
            </label>
          </div>
        </div>

        {/* Options Section */}
        <div className="mt-8 space-y-4">
          <div className="section-label">Output Options</div>
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              className="checkbox-card"
              data-selected={options.enableJerseyDetection}
              onClick={() => setOptions({ ...options, enableJerseyDetection: !options.enableJerseyDetection })}
            >
              <Users size={20} className="text-baseline-500" />
              <span className="text-sm font-medium">Player names</span>
            </button>
            <button
              type="button"
              className="checkbox-card"
              data-selected={options.enableTacticalView}
              onClick={() => setOptions({ ...options, enableTacticalView: !options.enableTacticalView })}
            >
              <Eye size={20} className="text-baseline-500" />
              <span className="text-sm font-medium">Court overview</span>
            </button>
            <button
              type="button"
              className="checkbox-card"
              data-selected={options.enableSegmentation}
              onClick={() => setOptions({ ...options, enableSegmentation: !options.enableSegmentation })}
            >
              <Film size={20} className="text-baseline-500" />
              <span className="text-sm font-medium">Player outlines</span>
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">
            {error}
          </div>
        )}

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          disabled={isSubmitting || !video}
          className="w-full mt-8 btn-primary text-base py-4"
        >
          {isSubmitting ? 'Uploading...' : 'Analyze Video'}
        </button>

        <p className="mt-4 text-center text-sm text-baseline-500">
          Processing takes 2-5 minutes. We&apos;ll notify you when ready.
        </p>
      </main>
    </div>
  );
}

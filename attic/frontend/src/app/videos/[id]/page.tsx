'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import {
 ArrowLeft,
 Download,
 RefreshCw,
 Loader,
 CheckCircle,
 XCircle,
 Clock,
 Play,
 Pause,
 Volume2,
 VolumeX,
 Maximize,
 Users,
 Eye,
 Film,
} from 'lucide-react';
import { jobsApi, JobDetail, VisualizationConfig } from '@/lib/api';
import { cn, formatDate, formatDuration } from '@/lib/utils';

// Header component
function Header() {
 return (
 <header className="flex items-center justify-between px-8 py-4 border-b border-baseline-100 bg-white">
 <Link href="/" className="flex items-center gap-2">
 <span className="text-2xl font-bold text-blacktop">
 SWISH<span className="text-court">/</span>VISION
 </span>
 </Link>
 <nav className="flex items-center gap-6">
 <Link href="/videos" className="text-sm font-medium text-baseline-600 hover:text-blacktop transition-colors">
 My Videos
 </Link>
 <button className="btn-secondary text-sm py-2 px-4">Sign In</button>
 </nav>
 </header>
 );
}

// Processing stage labels
const STAGE_LABELS: Record<string, string> = {
 detecting: 'Detecting players...',
 tracking: 'Tracking players...',
 segmenting: 'Creating player outlines...',
 classifying: 'Identifying teams...',
 reading_jerseys: 'Reading jersey numbers...',
 rendering: 'Rendering final video...',
};

// Status indicator component
function StatusIndicator({ status, progress, currentStage }: { status: string; progress: number; currentStage?: string }) {
 if (status === 'completed') {
 return (
 <div className="flex items-center gap-3 text-green-600">
 <CheckCircle size={24} />
 <span className="font-medium">Analysis Complete</span>
 </div>
 );
 }

 if (status === 'failed') {
 return (
 <div className="flex items-center gap-3 text-red-600">
 <XCircle size={24} />
 <span className="font-medium">Analysis Failed</span>
 </div>
 );
 }

 if (status === 'cancelled') {
 return (
 <div className="flex items-center gap-3 text-baseline-500">
 <XCircle size={24} />
 <span className="font-medium">Cancelled</span>
 </div>
 );
 }

 const isProcessing = ['processing', 'rendering', 'queued', 'uploading', 'pending'].includes(status);

 if (isProcessing) {
 return (
 <div className="space-y-3">
 <div className="flex items-center gap-3 text-court">
 <Loader size={24} className="animate-spin" />
 <span className="font-medium">
 {currentStage ? STAGE_LABELS[currentStage] || 'Processing...' : 'Processing...'}
 </span>
 </div>
 <div className="w-full max-w-md">
 <div className="h-2 bg-baseline-100 rounded-full overflow-hidden">
 <div
 className="h-full bg-court transition-all duration-500"
 style={{ width: `${progress}%` }}
 />
 </div>
 <p className="text-sm text-baseline-500 mt-2">{progress}% complete</p>
 </div>
 </div>
 );
 }

 return null;
}

// Video player component
function VideoPlayer({ url, poster }: { url: string; poster?: string }) {
 const videoRef = useRef<HTMLVideoElement>(null);
 const [isPlaying, setIsPlaying] = useState(false);
 const [isMuted, setIsMuted] = useState(false);
 const [progress, setProgress] = useState(0);
 const [duration, setDuration] = useState(0);

 const togglePlay = () => {
 if (videoRef.current) {
 if (isPlaying) {
 videoRef.current.pause();
 } else {
 videoRef.current.play();
 }
 setIsPlaying(!isPlaying);
 }
 };

 const toggleMute = () => {
 if (videoRef.current) {
 videoRef.current.muted = !isMuted;
 setIsMuted(!isMuted);
 }
 };

 const handleTimeUpdate = () => {
 if (videoRef.current) {
 setProgress((videoRef.current.currentTime / videoRef.current.duration) * 100);
 }
 };

 const handleLoadedMetadata = () => {
 if (videoRef.current) {
 setDuration(videoRef.current.duration);
 }
 };

 const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
 if (videoRef.current) {
 const rect = e.currentTarget.getBoundingClientRect();
 const percent = (e.clientX - rect.left) / rect.width;
 videoRef.current.currentTime = percent * videoRef.current.duration;
 }
 };

 const handleFullscreen = () => {
 if (videoRef.current) {
 videoRef.current.requestFullscreen();
 }
 };

 const formatTime = (seconds: number) => {
 const mins = Math.floor(seconds / 60);
 const secs = Math.floor(seconds % 60);
 return `${mins}:${secs.toString().padStart(2, '0')}`;
 };

 return (
 <div className="relative bg-blacktop rounded-xl overflow-hidden">
 <video
 ref={videoRef}
 src={url}
 poster={poster}
 className="w-full aspect-video"
 onTimeUpdate={handleTimeUpdate}
 onLoadedMetadata={handleLoadedMetadata}
 onPlay={() => setIsPlaying(true)}
 onPause={() => setIsPlaying(false)}
 onEnded={() => setIsPlaying(false)}
 />

 {/* Controls overlay */}
 <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-4">
 {/* Progress bar */}
 <div
 className="h-1 bg-white/30 rounded-full cursor-pointer mb-3"
 onClick={handleSeek}
 >
 <div
 className="h-full bg-court transition-all"
 style={{ width: `${progress}%` }}
 />
 </div>

 {/* Controls */}
 <div className="flex items-center justify-between">
 <div className="flex items-center gap-3">
 <button
 onClick={togglePlay}
 className="w-10 h-10 bg-white/20 hover:bg-white/30 rounded-full flex items-center justify-center transition-colors"
 >
 {isPlaying ? <Pause size={20} className="text-white" /> : <Play size={20} className="text-white ml-0.5" />}
 </button>
 <button
 onClick={toggleMute}
 className="w-8 h-8 bg-white/10 hover:bg-white/20 rounded-full flex items-center justify-center transition-colors"
 >
 {isMuted ? <VolumeX size={16} className="text-white" /> : <Volume2 size={16} className="text-white" />}
 </button>
 <span className="text-sm text-white/80">
 {formatTime((progress / 100) * duration)} / {formatTime(duration)}
 </span>
 </div>
 <button
 onClick={handleFullscreen}
 className="w-8 h-8 bg-white/10 hover:bg-white/20 rounded-full flex items-center justify-center transition-colors"
 >
 <Maximize size={16} className="text-white" />
 </button>
 </div>
 </div>

 {/* Play button overlay when paused */}
 {!isPlaying && (
 <button
 onClick={togglePlay}
 className="absolute inset-0 flex items-center justify-center bg-black/30 hover:bg-black/40 transition-colors"
 >
 <div className="w-20 h-20 bg-white/90 rounded-full flex items-center justify-center">
 <Play size={36} className="text-blacktop ml-1" />
 </div>
 </button>
 )}
 </div>
 );
}

// Players list component
function PlayersList({ players, teamA, teamB }: { players: JobDetail['detected_players']; teamA: any; teamB: any }) {
 if (!players || players.length === 0) {
 return (
 <div className="text-center py-8 text-baseline-500">
 No players detected
 </div>
 );
 }

 const teamAPlayers = players.filter((p) => p.team === 'team_a');
 const teamBPlayers = players.filter((p) => p.team === 'team_b');

 const TeamSection = ({ teamName, color, players }: { teamName: string; color: string; players: typeof teamAPlayers }) => (
 <div>
 <div className="flex items-center gap-2 mb-3">
 <div className="w-4 h-4 rounded " style={{ backgroundColor: color }} />
 <h4 className="font-medium text-blacktop">{teamName}</h4>
 <span className="text-sm text-baseline-500">({players.length})</span>
 </div>
 <div className="space-y-2">
 {players.map((player) => (
 <div
 key={player.id}
 className="flex items-center justify-between p-3 bg-baseline-50 rounded-lg "
 >
 <div className="flex items-center gap-3">
 <span className="w-8 h-8 bg-white border border-baseline-200 rounded flex items-center justify-center text-sm font-medium">
 {player.jersey_number || '?'}
 </span>
 <span className="text-sm">{player.player_name || `Player ${player.id}`}</span>
 </div>
 <span className="text-xs text-baseline-500">{player.frames_tracked} frames</span>
 </div>
 ))}
 </div>
 </div>
 );

 return (
 <div className="grid grid-cols-2 gap-6">
 <TeamSection teamName={teamA.name || 'Team A'} color={teamA.color} players={teamAPlayers} />
 <TeamSection teamName={teamB.name || 'Team B'} color={teamB.color} players={teamBPlayers} />
 </div>
 );
}

export default function VideoDetailPage() {
 const params = useParams();
 const router = useRouter();
 const jobId = params.id as string;

 const [job, setJob] = useState<JobDetail | null>(null);
 const [isLoading, setIsLoading] = useState(true);
 const [error, setError] = useState<string | null>(null);
 const [activeTab, setActiveTab] = useState<'video' | 'players'>('video');

 // Fetch job details
 useEffect(() => {
 const fetchJob = async () => {
 setIsLoading(true);
 setError(null);
 try {
 const data = await jobsApi.get(jobId);
 setJob(data);
 } catch (err) {
 setError(err instanceof Error ? err.message : 'Failed to load video');
 } finally {
 setIsLoading(false);
 }
 };

 if (jobId) {
 fetchJob();
 }
 }, [jobId]);

 // Poll for updates while processing
 useEffect(() => {
 if (!job) return;

 const isProcessing = ['pending', 'uploading', 'queued', 'processing', 'rendering'].includes(job.status);
 if (!isProcessing) return;

 const interval = setInterval(async () => {
 try {
 const data = await jobsApi.get(jobId);
 setJob(data);
 } catch {
 // Silently fail polling
 }
 }, 3000);

 return () => clearInterval(interval);
 }, [job, jobId]);

 if (isLoading) {
 return (
 <div className="min-h-screen bg-sideline">
 <Header />
 <div className="flex items-center justify-center py-32">
 <Loader size={32} className="animate-spin text-court" />
 </div>
 </div>
 );
 }

 if (error || !job) {
 return (
 <div className="min-h-screen bg-sideline">
 <Header />
 <main className="w-full px-8 lg:px-16 py-12">
 <div className="text-center py-16 bg-white border border-baseline-100">
 <XCircle size={48} className="mx-auto text-red-400 mb-4" />
 <h2 className="text-lg font-medium text-blacktop mb-2">Error Loading Video</h2>
 <p className="text-baseline-500 mb-6">{error || 'Video not found'}</p>
 <Link href="/videos" className="btn-secondary py-2.5 px-6">
 Back to Videos
 </Link>
 </div>
 </main>
 </div>
 );
 }

 const isCompleted = job.status === 'completed';
 const finalVideoUrl = job.video_urls?.final;

 return (
 <div className="min-h-screen bg-sideline">
 <Header />

 <main className="w-full px-8 lg:px-16 py-8">
 {/* Back link */}
 <Link
 href="/videos"
 className="inline-flex items-center gap-2 text-sm text-baseline-600 hover:text-blacktop mb-6"
 >
 <ArrowLeft size={16} />
 Back to Videos
 </Link>

 {/* Title and meta */}
 <div className="flex items-start justify-between mb-6">
 <div>
 <h1 className="text-2xl font-bold text-blacktop">
 {job.team_a.name && job.team_b.name
 ? `${job.team_a.name} vs ${job.team_b.name}`
 : job.input_video_filename || 'Video Analysis'}
 </h1>
 <div className="flex items-center gap-4 mt-2 text-sm text-baseline-500">
 <span>{formatDate(job.created_at)}</span>
 {job.input_video_duration_seconds && (
 <span>{formatDuration(job.input_video_duration_seconds)}</span>
 )}
 {job.processing_time_seconds && isCompleted && (
 <span>Processed in {formatDuration(job.processing_time_seconds)}</span>
 )}
 </div>
 </div>

 {isCompleted && finalVideoUrl && (
 <a
 href={finalVideoUrl}
 download
 className="btn-primary py-2.5 px-5 flex items-center gap-2"
 >
 <Download size={18} />
 Download
 </a>
 )}
 </div>

 {/* Status indicator for non-completed jobs */}
 {!isCompleted && (
 <div className="bg-white border border-baseline-100 rounded-xl p-6 mb-6">
 <StatusIndicator
 status={job.status}
 progress={job.progress}
 currentStage={job.current_stage || undefined}
 />
 {job.error_message && (
 <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">
 {job.error_message}
 </div>
 )}
 </div>
 )}

 {/* Main content area */}
 {isCompleted && (
 <>
 {/* Tabs */}
 <div className="flex items-center gap-1 mb-4 bg-baseline-100 rounded-lg p-1 w-fit">
 <button
 onClick={() => setActiveTab('video')}
 className={cn(
 'px-4 py-2 rounded-md text-sm font-medium transition-colors',
 activeTab === 'video' ? 'bg-white text-blacktop shadow-sm' : 'text-baseline-600 hover:text-blacktop'
 )}
 >
 <span className="flex items-center gap-2">
 <Film size={16} />
 Video
 </span>
 </button>
 <button
 onClick={() => setActiveTab('players')}
 className={cn(
 'px-4 py-2 rounded-md text-sm font-medium transition-colors',
 activeTab === 'players' ? 'bg-white text-blacktop shadow-sm' : 'text-baseline-600 hover:text-blacktop'
 )}
 >
 <span className="flex items-center gap-2">
 <Users size={16} />
 Players ({job.detected_players?.length || 0})
 </span>
 </button>
 </div>

 {/* Tab content */}
 <div className="bg-white border border-baseline-100 rounded-xl overflow-hidden">
 {activeTab === 'video' && finalVideoUrl && (
 <VideoPlayer url={finalVideoUrl} />
 )}

 {activeTab === 'players' && (
 <div className="p-6">
 <PlayersList
 players={job.detected_players}
 teamA={job.team_a}
 teamB={job.team_b}
 />
 </div>
 )}
 </div>

 {/* Settings summary */}
 <div className="mt-6 bg-white border border-baseline-100 rounded-xl p-6">
 <h3 className="font-medium text-blacktop mb-4">Analysis Settings</h3>
 <div className="grid grid-cols-3 gap-4">
 <div className="flex items-center gap-3">
 <div className={cn(
 'w-8 h-8 rounded-lg flex items-center justify-center',
 job.processing_config.enable_segmentation ? 'bg-green-100' : 'bg-baseline-100'
 )}>
 <Film size={16} className={job.processing_config.enable_segmentation ? 'text-green-600' : 'text-baseline-400'} />
 </div>
 <div>
 <p className="text-sm font-medium">Player Outlines</p>
 <p className="text-xs text-baseline-500">
 {job.processing_config.enable_segmentation ? 'Enabled' : 'Disabled'}
 </p>
 </div>
 </div>
 <div className="flex items-center gap-3">
 <div className={cn(
 'w-8 h-8 rounded-lg flex items-center justify-center',
 job.processing_config.enable_jersey_detection ? 'bg-green-100' : 'bg-baseline-100'
 )}>
 <Users size={16} className={job.processing_config.enable_jersey_detection ? 'text-green-600' : 'text-baseline-400'} />
 </div>
 <div>
 <p className="text-sm font-medium">Jersey Detection</p>
 <p className="text-xs text-baseline-500">
 {job.processing_config.enable_jersey_detection ? 'Enabled' : 'Disabled'}
 </p>
 </div>
 </div>
 <div className="flex items-center gap-3">
 <div className={cn(
 'w-8 h-8 rounded-lg flex items-center justify-center',
 job.processing_config.enable_tactical_view ? 'bg-green-100' : 'bg-baseline-100'
 )}>
 <Eye size={16} className={job.processing_config.enable_tactical_view ? 'text-green-600' : 'text-baseline-400'} />
 </div>
 <div>
 <p className="text-sm font-medium">Court Overview</p>
 <p className="text-xs text-baseline-500">
 {job.processing_config.enable_tactical_view ? 'Enabled' : 'Disabled'}
 </p>
 </div>
 </div>
 </div>
 </div>
 </>
 )}
 </main>
 </div>
 );
}

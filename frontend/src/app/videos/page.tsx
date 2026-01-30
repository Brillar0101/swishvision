'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Film, Clock, CheckCircle, XCircle, Loader, ChevronLeft, ChevronRight } from 'lucide-react';
import { jobsApi, JobSummary } from '@/lib/api';
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
        <Link href="/videos" className="text-sm font-medium text-court">
          My Videos
        </Link>
        <button className="btn-secondary text-sm py-2 px-4">
          Sign In
        </button>
      </nav>
    </header>
  );
}

// Status badge component
function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { icon: React.ReactNode; label: string; className: string }> = {
    pending: {
      icon: <Clock size={14} />,
      label: 'Pending',
      className: 'bg-baseline-100 text-baseline-600',
    },
    uploading: {
      icon: <Loader size={14} className="animate-spin" />,
      label: 'Uploading',
      className: 'bg-blue-100 text-blue-700',
    },
    queued: {
      icon: <Clock size={14} />,
      label: 'Queued',
      className: 'bg-baseline-100 text-baseline-600',
    },
    processing: {
      icon: <Loader size={14} className="animate-spin" />,
      label: 'Processing',
      className: 'bg-court-100 text-court-700',
    },
    rendering: {
      icon: <Loader size={14} className="animate-spin" />,
      label: 'Rendering',
      className: 'bg-court-100 text-court-700',
    },
    completed: {
      icon: <CheckCircle size={14} />,
      label: 'Completed',
      className: 'bg-green-100 text-green-700',
    },
    failed: {
      icon: <XCircle size={14} />,
      label: 'Failed',
      className: 'bg-red-100 text-red-700',
    },
    cancelled: {
      icon: <XCircle size={14} />,
      label: 'Cancelled',
      className: 'bg-baseline-100 text-baseline-600',
    },
  };

  const { icon, label, className } = config[status] || config.pending;

  return (
    <span className={cn('inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium', className)}>
      {icon}
      {label}
    </span>
  );
}

// Job card component
function JobCard({ job }: { job: JobSummary }) {
  const router = useRouter();

  return (
    <div
      onClick={() => router.push(`/videos/${job.id}`)}
      className="bg-white border border-baseline-200 rounded-xl p-5 hover:border-court hover:shadow-card-hover transition-all cursor-pointer"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 bg-baseline-50 rounded-lg flex items-center justify-center flex-shrink-0">
            <Film size={24} className="text-baseline-400" />
          </div>
          <div>
            <h3 className="font-medium text-blacktop">
              {job.team_a_name && job.team_b_name
                ? `${job.team_a_name} vs ${job.team_b_name}`
                : job.input_video_filename || 'Untitled Video'}
            </h3>
            <p className="text-sm text-baseline-500 mt-0.5">
              {job.input_video_filename}
            </p>
            <p className="text-xs text-baseline-400 mt-2">
              {formatDate(job.created_at)}
            </p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          <StatusBadge status={job.status} />
          {job.status === 'processing' && (
            <div className="w-24">
              <div className="h-1 bg-baseline-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-court transition-all duration-300"
                  style={{ width: `${job.progress}%` }}
                />
              </div>
              <p className="text-xs text-baseline-500 mt-1 text-right">{job.progress}%</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Filter tabs
const STATUS_FILTERS = [
  { label: 'All', value: '' },
  { label: 'Processing', value: 'processing' },
  { label: 'Completed', value: 'completed' },
  { label: 'Failed', value: 'failed' },
];

export default function VideosPage() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [statusFilter, setStatusFilter] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const perPage = 10;

  useEffect(() => {
    const fetchJobs = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await jobsApi.list(page, perPage, statusFilter || undefined);
        setJobs(response.jobs);
        setTotal(response.total);
        setTotalPages(response.total_pages);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load videos');
      } finally {
        setIsLoading(false);
      }
    };

    fetchJobs();
  }, [page, statusFilter]);

  // Poll for updates when there are processing jobs
  useEffect(() => {
    const hasProcessingJobs = jobs.some((job) =>
      ['pending', 'uploading', 'queued', 'processing', 'rendering'].includes(job.status)
    );

    if (!hasProcessingJobs) return;

    const interval = setInterval(async () => {
      try {
        const response = await jobsApi.list(page, perPage, statusFilter || undefined);
        setJobs(response.jobs);
        setTotal(response.total);
        setTotalPages(response.total_pages);
      } catch {
        // Silently fail polling
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [jobs, page, statusFilter]);

  return (
    <div className="min-h-screen bg-sideline">
      <Header />

      <main className="w-full px-8 lg:px-16 py-12">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-blacktop">My Videos</h1>
            <p className="text-baseline-500 mt-1">
              {total} video{total !== 1 ? 's' : ''} analyzed
            </p>
          </div>
          <Link href="/" className="btn-primary py-2.5 px-5">
            New Analysis
          </Link>
        </div>

        {/* Filter tabs */}
        <div className="flex items-center gap-2 mb-6">
          {STATUS_FILTERS.map((filter) => (
            <button
              key={filter.value}
              onClick={() => {
                setStatusFilter(filter.value);
                setPage(1);
              }}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                statusFilter === filter.value
                  ? 'bg-blacktop text-white'
                  : 'bg-white text-baseline-600 hover:bg-baseline-50 border border-baseline-300'
              )}
            >
              {filter.label}
            </button>
          ))}
        </div>

        {/* Error state */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600 mb-6">
            {error}
          </div>
        )}

        {/* Loading state */}
        {isLoading ? (
          <div className="flex items-center justify-center py-16">
            <Loader size={32} className="animate-spin text-court" />
          </div>
        ) : jobs.length === 0 ? (
          /* Empty state */
          <div className="text-center py-16 bg-white border border-baseline-200 rounded-xl">
            <Film size={48} className="mx-auto text-baseline-300 mb-4" />
            <h2 className="text-lg font-medium text-blacktop mb-2">No videos yet</h2>
            <p className="text-baseline-500 mb-6">Upload your first video to get started</p>
            <Link href="/" className="btn-primary py-2.5 px-6">
              Upload Video
            </Link>
          </div>
        ) : (
          /* Job list */
          <div className="space-y-3">
            {jobs.map((job) => (
              <JobCard key={job.id} job={job} />
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2 mt-8">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-2 rounded-lg hover:bg-baseline-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft size={20} />
            </button>
            <span className="text-sm text-baseline-600">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="p-2 rounded-lg hover:bg-baseline-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronRight size={20} />
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

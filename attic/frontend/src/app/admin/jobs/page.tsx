'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import {
 Search,
 Filter,
 Loader,
 CheckCircle,
 XCircle,
 Clock,
 MoreHorizontal,
 ChevronLeft,
 ChevronRight,
 RefreshCw,
 Trash2,
 Eye,
 StopCircle,
} from 'lucide-react';
import { cn, formatDate, formatRelativeTime, formatDuration } from '@/lib/utils';

// Status badge component
function StatusBadge({ status }: { status: string }) {
 const config: Record<string, { icon: React.ReactNode; label: string; className: string }> = {
 pending: {
 icon: <Clock size={12} />,
 label: 'Pending',
 className: 'bg-baseline-100 text-baseline-600',
 },
 uploading: {
 icon: <Loader size={12} className="animate-spin" />,
 label: 'Uploading',
 className: 'bg-blue-100 text-blue-700',
 },
 queued: {
 icon: <Clock size={12} />,
 label: 'Queued',
 className: 'bg-baseline-100 text-baseline-600',
 },
 processing: {
 icon: <Loader size={12} className="animate-spin" />,
 label: 'Processing',
 className: 'bg-court-100 text-court-700',
 },
 rendering: {
 icon: <Loader size={12} className="animate-spin" />,
 label: 'Rendering',
 className: 'bg-court-100 text-court-700',
 },
 completed: {
 icon: <CheckCircle size={12} />,
 label: 'Completed',
 className: 'bg-green-100 text-green-700',
 },
 failed: {
 icon: <XCircle size={12} />,
 label: 'Failed',
 className: 'bg-red-100 text-red-700',
 },
 cancelled: {
 icon: <XCircle size={12} />,
 label: 'Cancelled',
 className: 'bg-baseline-100 text-baseline-600',
 },
 };

 const { icon, label, className } = config[status] || config.pending;

 return (
 <span className={cn('inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium', className)}>
 {icon}
 {label}
 </span>
 );
}

// Job action menu
function JobActionsMenu({ jobId, status, onAction }: { jobId: string; status: string; onAction: (action: string, id: string) => void }) {
 const [isOpen, setIsOpen] = useState(false);

 const canCancel = ['pending', 'queued', 'processing'].includes(status);
 const canRetry = ['failed', 'cancelled'].includes(status);

 return (
 <div className="relative">
 <button
 onClick={() => setIsOpen(!isOpen)}
 className="p-1.5 rounded-lg hover:bg-baseline-100 transition-colors"
 >
 <MoreHorizontal size={16} className="text-baseline-500" />
 </button>

 {isOpen && (
 <>
 <div className="fixed inset-0 z-10" onClick={() => setIsOpen(false)} />
 <div className="absolute right-0 top-full mt-1 w-40 bg-white shadow-lg border border-baseline-100 rounded-lg py-1 z-20">
 <Link
 href={`/videos/${jobId}`}
 className="flex items-center gap-2 px-3 py-2 text-sm text-baseline-600 hover:bg-baseline-50 transition-colors"
 onClick={() => setIsOpen(false)}
 >
 <Eye size={14} />
 View Details
 </Link>
 {canCancel && (
 <button
 onClick={() => { onAction('cancel', jobId); setIsOpen(false); }}
 className="w-full flex items-center gap-2 px-3 py-2 text-sm text-baseline-600 hover:bg-baseline-50 transition-colors"
 >
 <StopCircle size={14} />
 Cancel Job
 </button>
 )}
 {canRetry && (
 <button
 onClick={() => { onAction('retry', jobId); setIsOpen(false); }}
 className="w-full flex items-center gap-2 px-3 py-2 text-sm text-baseline-600 hover:bg-baseline-50 transition-colors"
 >
 <RefreshCw size={14} />
 Retry Job
 </button>
 )}
 <button
 onClick={() => { onAction('delete', jobId); setIsOpen(false); }}
 className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
 >
 <Trash2 size={14} />
 Delete
 </button>
 </div>
 </>
 )}
 </div>
 );
}

// Mock data
interface AdminJob {
 id: string;
 teamA: string;
 teamB: string;
 status: string;
 progress: number;
 createdAt: string;
 completedAt?: string;
 processingTime?: number;
 user: {
 email: string;
 name?: string;
 };
 filename: string;
 error?: string;
}

const MOCK_JOBS: AdminJob[] = [
 { id: 'job-001', teamA: 'Lakers', teamB: 'Celtics', status: 'processing', progress: 65, createdAt: new Date(Date.now() - 300000).toISOString(), user: { email: 'john@example.com', name: 'John Doe' }, filename: 'lakers_celtics_q4.mp4' },
 { id: 'job-002', teamA: 'Warriors', teamB: 'Heat', status: 'completed', progress: 100, createdAt: new Date(Date.now() - 1800000).toISOString(), completedAt: new Date(Date.now() - 1500000).toISOString(), processingTime: 312, user: { email: 'jane@example.com', name: 'Jane Smith' }, filename: 'warriors_heat_full.mp4' },
 { id: 'job-003', teamA: 'Pacers', teamB: 'Thunder', status: 'completed', progress: 100, createdAt: new Date(Date.now() - 3600000).toISOString(), completedAt: new Date(Date.now() - 3300000).toISOString(), processingTime: 245, user: { email: 'mike@example.com' }, filename: 'pacers_okc.mp4' },
 { id: 'job-004', teamA: 'Bucks', teamB: 'Suns', status: 'failed', progress: 42, createdAt: new Date(Date.now() - 7200000).toISOString(), user: { email: 'sarah@example.com', name: 'Sarah Wilson' }, filename: 'bucks_suns_highlights.mov', error: 'GPU memory exceeded during segmentation' },
 { id: 'job-005', teamA: 'Nuggets', teamB: 'Mavs', status: 'queued', progress: 0, createdAt: new Date(Date.now() - 900000).toISOString(), user: { email: 'alex@example.com' }, filename: 'nuggets_mavs_q2.mp4' },
 { id: 'job-006', teamA: 'Knicks', teamB: '76ers', status: 'pending', progress: 0, createdAt: new Date(Date.now() - 600000).toISOString(), user: { email: 'chris@example.com', name: 'Chris Johnson' }, filename: 'knicks_sixers.mp4' },
 { id: 'job-007', teamA: 'Clippers', teamB: 'Kings', status: 'completed', progress: 100, createdAt: new Date(Date.now() - 86400000).toISOString(), completedAt: new Date(Date.now() - 86100000).toISOString(), processingTime: 198, user: { email: 'emma@example.com' }, filename: 'clippers_kings_ot.mp4' },
 { id: 'job-008', teamA: 'Nets', teamB: 'Bulls', status: 'cancelled', progress: 0, createdAt: new Date(Date.now() - 172800000).toISOString(), user: { email: 'david@example.com', name: 'David Brown' }, filename: 'nets_bulls.mp4' },
];

const STATUS_FILTERS = [
 { label: 'All', value: '' },
 { label: 'Processing', value: 'processing' },
 { label: 'Queued', value: 'queued' },
 { label: 'Completed', value: 'completed' },
 { label: 'Failed', value: 'failed' },
 { label: 'Cancelled', value: 'cancelled' },
];

export default function AdminJobsPage() {
 const [jobs, setJobs] = useState<AdminJob[]>([]);
 const [isLoading, setIsLoading] = useState(true);
 const [searchQuery, setSearchQuery] = useState('');
 const [statusFilter, setStatusFilter] = useState('');
 const [page, setPage] = useState(1);
 const [selectedJobs, setSelectedJobs] = useState<Set<string>>(new Set());

 const perPage = 10;
 const totalPages = Math.ceil(MOCK_JOBS.length / perPage);

 // Simulate loading
 useEffect(() => {
 const timer = setTimeout(() => {
 setJobs(MOCK_JOBS);
 setIsLoading(false);
 }, 500);
 return () => clearTimeout(timer);
 }, []);

 // Filter jobs
 const filteredJobs = jobs.filter((job) => {
 if (statusFilter && job.status !== statusFilter) return false;
 if (searchQuery) {
 const query = searchQuery.toLowerCase();
 return (
 job.teamA.toLowerCase().includes(query) ||
 job.teamB.toLowerCase().includes(query) ||
 job.user.email.toLowerCase().includes(query) ||
 job.filename.toLowerCase().includes(query)
 );
 }
 return true;
 });

 const handleJobAction = (action: string, jobId: string) => {
 console.log(`Action: ${action} on job: ${jobId}`);
 // In production, this would call the API
 };

 const toggleJobSelection = (jobId: string) => {
 const newSelection = new Set(selectedJobs);
 if (newSelection.has(jobId)) {
 newSelection.delete(jobId);
 } else {
 newSelection.add(jobId);
 }
 setSelectedJobs(newSelection);
 };

 const toggleAllSelection = () => {
 if (selectedJobs.size === filteredJobs.length) {
 setSelectedJobs(new Set());
 } else {
 setSelectedJobs(new Set(filteredJobs.map((j) => j.id)));
 }
 };

 if (isLoading) {
 return (
 <div className="flex items-center justify-center py-32">
 <Loader size={32} className="animate-spin text-court" />
 </div>
 );
 }

 return (
 <div>
 <div className="mb-6">
 <h1 className="text-2xl font-bold text-blacktop">Jobs</h1>
 <p className="text-baseline-500 mt-1">Manage video analysis jobs</p>
 </div>

 {/* Filters bar */}
 <div className="flex items-center justify-between gap-4 mb-6">
 <div className="flex items-center gap-3">
 {/* Search */}
 <div className="relative">
 <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-baseline-400" />
 <input
 type="text"
 placeholder="Search jobs..."
 value={searchQuery}
 onChange={(e) => setSearchQuery(e.target.value)}
 className="w-64 pl-9 pr-4 py-2 rounded-lg border border-baseline-200 text-sm focus:outline-none focus:border-court"
 />
 </div>

 {/* Status filter */}
 <select
 value={statusFilter}
 onChange={(e) => setStatusFilter(e.target.value)}
 className="px-3 py-2 rounded-lg border border-baseline-200 text-sm focus:outline-none focus:border-court bg-white"
 >
 {STATUS_FILTERS.map((filter) => (
 <option key={filter.value} value={filter.value}>
 {filter.label}
 </option>
 ))}
 </select>
 </div>

 {/* Bulk actions */}
 {selectedJobs.size > 0 && (
 <div className="flex items-center gap-2">
 <span className="text-sm text-baseline-500">{selectedJobs.size} selected</span>
 <button className="btn-secondary text-sm py-1.5 px-3">
 Cancel Selected
 </button>
 <button className="btn-secondary text-sm py-1.5 px-3 text-red-600 border-red-200 hover:bg-red-50">
 Delete Selected
 </button>
 </div>
 )}
 </div>

 {/* Jobs table */}
 <div className="bg-white border border-baseline-100 rounded-xl overflow-hidden">
 <table className="w-full">
 <thead>
 <tr className="bg-baseline-50 text-left text-xs text-baseline-500 uppercase tracking-wider">
 <th className="px-4 py-3 w-8">
 <input
 type="checkbox"
 checked={selectedJobs.size === filteredJobs.length && filteredJobs.length > 0}
 onChange={toggleAllSelection}
 className="rounded border-baseline-300"
 />
 </th>
 <th className="px-4 py-3">Job</th>
 <th className="px-4 py-3">User</th>
 <th className="px-4 py-3">Status</th>
 <th className="px-4 py-3">Created</th>
 <th className="px-4 py-3">Duration</th>
 <th className="px-4 py-3 w-12"></th>
 </tr>
 </thead>
 <tbody className="divide-y divide-baseline-100">
 {filteredJobs.map((job) => (
 <tr key={job.id} className={cn('hover:bg-baseline-50 transition-colors', selectedJobs.has(job.id) && 'bg-court-50')}>
 <td className="px-4 py-3">
 <input
 type="checkbox"
 checked={selectedJobs.has(job.id)}
 onChange={() => toggleJobSelection(job.id)}
 className="rounded border-baseline-300"
 />
 </td>
 <td className="px-4 py-3">
 <div>
 <Link href={`/videos/${job.id}`} className="font-medium text-blacktop hover:text-court">
 {job.teamA} vs {job.teamB}
 </Link>
 <p className="text-xs text-baseline-500 mt-0.5">{job.filename}</p>
 </div>
 </td>
 <td className="px-4 py-3">
 <div>
 <p className="text-sm text-blacktop">{job.user.name || job.user.email.split('@')[0]}</p>
 <p className="text-xs text-baseline-500">{job.user.email}</p>
 </div>
 </td>
 <td className="px-4 py-3">
 <div className="flex items-center gap-2">
 <StatusBadge status={job.status} />
 {job.status === 'processing' && (
 <span className="text-xs text-baseline-500">{job.progress}%</span>
 )}
 </div>
 {job.error && (
 <p className="text-xs text-red-600 mt-1 max-w-xs truncate" title={job.error}>
 {job.error}
 </p>
 )}
 </td>
 <td className="px-4 py-3 text-sm text-baseline-600">
 {formatRelativeTime(job.createdAt)}
 </td>
 <td className="px-4 py-3 text-sm text-baseline-600">
 {job.processingTime ? formatDuration(job.processingTime) : '-'}
 </td>
 <td className="px-4 py-3">
 <JobActionsMenu jobId={job.id} status={job.status} onAction={handleJobAction} />
 </td>
 </tr>
 ))}
 </tbody>
 </table>

 {filteredJobs.length === 0 && (
 <div className="text-center py-12 text-baseline-500">
 No jobs found matching your criteria
 </div>
 )}
 </div>

 {/* Pagination */}
 {totalPages > 1 && (
 <div className="flex items-center justify-between mt-4">
 <p className="text-sm text-baseline-500">
 Showing {(page - 1) * perPage + 1} to {Math.min(page * perPage, filteredJobs.length)} of {filteredJobs.length} jobs
 </p>
 <div className="flex items-center gap-2">
 <button
 onClick={() => setPage((p) => Math.max(1, p - 1))}
 disabled={page === 1}
 className="p-2 rounded-lg hover:bg-baseline-100 disabled:opacity-50 disabled:cursor-not-allowed"
 >
 <ChevronLeft size={18} />
 </button>
 <span className="text-sm text-baseline-600">
 Page {page} of {totalPages}
 </span>
 <button
 onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
 disabled={page === totalPages}
 className="p-2 rounded-lg hover:bg-baseline-100 disabled:opacity-50 disabled:cursor-not-allowed"
 >
 <ChevronRight size={18} />
 </button>
 </div>
 </div>
 )}
 </div>
 );
}

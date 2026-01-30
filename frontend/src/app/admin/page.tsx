'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import {
 Film,
 Users,
 Clock,
 CheckCircle,
 XCircle,
 AlertTriangle,
 TrendingUp,
 Activity,
 Loader,
} from 'lucide-react';
import { cn, formatRelativeTime } from '@/lib/utils';

// Stats card component
interface StatCardProps {
 label: string;
 value: string | number;
 icon: React.ReactNode;
 trend?: { value: number; isUp: boolean };
 color?: 'default' | 'green' | 'red' | 'yellow';
}

function StatCard({ label, value, icon, trend, color = 'default' }: StatCardProps) {
 const colorClasses = {
 default: 'bg-white',
 green: 'bg-green-50 border-green-200',
 red: 'bg-red-50 border-red-200',
 yellow: 'bg-yellow-50 border-yellow-200',
 };

 return (
 <div className={cn(' border border-baseline-100 rounded-xl p-5', colorClasses[color])}>
 <div className="flex items-start justify-between">
 <div>
 <p className="text-sm text-baseline-500">{label}</p>
 <p className="text-2xl font-bold text-blacktop mt-1">{value}</p>
 {trend && (
 <p className={cn('text-xs mt-2', trend.isUp ? 'text-green-600' : 'text-red-600')}>
 {trend.isUp ? '+' : '-'}{Math.abs(trend.value)}% from last period
 </p>
 )}
 </div>
 <div className="w-10 h-10 bg-baseline-100 rounded-lg flex items-center justify-center text-baseline-600">
 {icon}
 </div>
 </div>
 </div>
 );
}

// Recent jobs table component
interface RecentJob {
 id: string;
 teamA: string;
 teamB: string;
 status: string;
 progress: number;
 createdAt: string;
 user: string;
}

function RecentJobsTable({ jobs }: { jobs: RecentJob[] }) {
 const statusConfig: Record<string, { icon: React.ReactNode; className: string }> = {
 completed: { icon: <CheckCircle size={14} />, className: 'text-green-600' },
 processing: { icon: <Loader size={14} className="animate-spin" />, className: 'text-court' },
 failed: { icon: <XCircle size={14} />, className: 'text-red-600' },
 pending: { icon: <Clock size={14} />, className: 'text-baseline-500' },
 queued: { icon: <Clock size={14} />, className: 'text-baseline-500' },
 };

 return (
 <div className="bg-white border border-baseline-100 rounded-xl overflow-hidden">
 <div className="px-5 py-4 border-b border-baseline-100 flex items-center justify-between">
 <h3 className="font-medium text-blacktop">Recent Jobs</h3>
 <Link href="/admin/jobs" className="text-sm text-court hover:underline">
 View all
 </Link>
 </div>
 <table className="w-full">
 <thead>
 <tr className="bg-baseline-50 text-left text-xs text-baseline-500 uppercase tracking-wider">
 <th className="px-5 py-3">Match</th>
 <th className="px-5 py-3">User</th>
 <th className="px-5 py-3">Status</th>
 <th className="px-5 py-3">Created</th>
 </tr>
 </thead>
 <tbody className="divide-y divide-baseline-100">
 {jobs.map((job) => {
 const config = statusConfig[job.status] || statusConfig.pending;
 return (
 <tr key={job.id} className="hover:bg-baseline-50 transition-colors">
 <td className="px-5 py-4">
 <Link href={`/admin/jobs?id=${job.id}`} className="font-medium text-blacktop hover:text-court">
 {job.teamA} vs {job.teamB}
 </Link>
 </td>
 <td className="px-5 py-4 text-sm text-baseline-600">{job.user}</td>
 <td className="px-5 py-4">
 <span className={cn('inline-flex items-center gap-1.5 text-sm', config.className)}>
 {config.icon}
 <span className="capitalize">{job.status}</span>
 {job.status === 'processing' && (
 <span className="text-xs text-baseline-500">({job.progress}%)</span>
 )}
 </span>
 </td>
 <td className="px-5 py-4 text-sm text-baseline-500">{formatRelativeTime(job.createdAt)}</td>
 </tr>
 );
 })}
 </tbody>
 </table>
 </div>
 );
}

// System alerts component
interface SystemAlert {
 id: string;
 type: 'error' | 'warning' | 'info';
 message: string;
 timestamp: string;
}

function SystemAlerts({ alerts }: { alerts: SystemAlert[] }) {
 const alertConfig = {
 error: { icon: <XCircle size={16} />, className: 'bg-red-50 border-red-200 text-red-700' },
 warning: { icon: <AlertTriangle size={16} />, className: 'bg-yellow-50 border-yellow-200 text-yellow-700' },
 info: { icon: <Activity size={16} />, className: 'bg-blue-50 border-blue-200 text-blue-700' },
 };

 if (alerts.length === 0) {
 return (
 <div className="bg-white border border-baseline-100 rounded-xl p-5">
 <h3 className="font-medium text-blacktop mb-4">System Status</h3>
 <div className="flex items-center gap-3 text-green-600">
 <CheckCircle size={20} />
 <span className="text-sm">All systems operational</span>
 </div>
 </div>
 );
 }

 return (
 <div className="bg-white border border-baseline-100 rounded-xl p-5">
 <h3 className="font-medium text-blacktop mb-4">System Alerts</h3>
 <div className="space-y-3">
 {alerts.map((alert) => {
 const config = alertConfig[alert.type];
 return (
 <div
 key={alert.id}
 className={cn('flex items-start gap-3 p-3 rounded-lg border', config.className)}
 >
 {config.icon}
 <div className="flex-1">
 <p className="text-sm">{alert.message}</p>
 <p className="text-xs opacity-70 mt-1">{formatRelativeTime(alert.timestamp)}</p>
 </div>
 </div>
 );
 })}
 </div>
 </div>
 );
}

// Mock data - in production this would come from the API
const MOCK_STATS = {
 totalJobs: 1247,
 processingJobs: 3,
 completedToday: 42,
 failedToday: 2,
 totalUsers: 856,
 activeUsers: 124,
 avgProcessingTime: '4:32',
 storageUsed: '2.4 TB',
};

const MOCK_RECENT_JOBS: RecentJob[] = [
 { id: '1', teamA: 'Lakers', teamB: 'Celtics', status: 'processing', progress: 65, createdAt: new Date(Date.now() - 300000).toISOString(), user: 'john@example.com' },
 { id: '2', teamA: 'Warriors', teamB: 'Heat', status: 'completed', progress: 100, createdAt: new Date(Date.now() - 1800000).toISOString(), user: 'jane@example.com' },
 { id: '3', teamA: 'Pacers', teamB: 'Thunder', status: 'completed', progress: 100, createdAt: new Date(Date.now() - 3600000).toISOString(), user: 'mike@example.com' },
 { id: '4', teamA: 'Bucks', teamB: 'Suns', status: 'failed', progress: 0, createdAt: new Date(Date.now() - 7200000).toISOString(), user: 'sarah@example.com' },
 { id: '5', teamA: 'Nuggets', teamB: 'Mavs', status: 'queued', progress: 0, createdAt: new Date(Date.now() - 900000).toISOString(), user: 'alex@example.com' },
];

const MOCK_ALERTS: SystemAlert[] = [];

export default function AdminDashboardPage() {
 const [isLoading, setIsLoading] = useState(true);
 const [stats, setStats] = useState(MOCK_STATS);
 const [recentJobs, setRecentJobs] = useState<RecentJob[]>([]);
 const [alerts, setAlerts] = useState<SystemAlert[]>([]);

 // Simulate loading
 useEffect(() => {
 const timer = setTimeout(() => {
 setRecentJobs(MOCK_RECENT_JOBS);
 setAlerts(MOCK_ALERTS);
 setIsLoading(false);
 }, 500);
 return () => clearTimeout(timer);
 }, []);

 if (isLoading) {
 return (
 <div className="flex items-center justify-center py-32">
 <Loader size={32} className="animate-spin text-court" />
 </div>
 );
 }

 return (
 <div>
 <div className="mb-8">
 <h1 className="text-2xl font-bold text-blacktop">Dashboard</h1>
 <p className="text-baseline-500 mt-1">Overview of your SwishVision instance</p>
 </div>

 {/* Stats grid */}
 <div className="grid grid-cols-4 gap-4 mb-8">
 <StatCard
 label="Total Jobs"
 value={stats.totalJobs.toLocaleString()}
 icon={<Film size={20} />}
 trend={{ value: 12, isUp: true }}
 />
 <StatCard
 label="Processing Now"
 value={stats.processingJobs}
 icon={<Activity size={20} />}
 color={stats.processingJobs > 5 ? 'yellow' : 'default'}
 />
 <StatCard
 label="Completed Today"
 value={stats.completedToday}
 icon={<CheckCircle size={20} />}
 color="green"
 />
 <StatCard
 label="Failed Today"
 value={stats.failedToday}
 icon={<XCircle size={20} />}
 color={stats.failedToday > 0 ? 'red' : 'default'}
 />
 </div>

 <div className="grid grid-cols-4 gap-4 mb-8">
 <StatCard
 label="Total Users"
 value={stats.totalUsers.toLocaleString()}
 icon={<Users size={20} />}
 />
 <StatCard
 label="Active Users (24h)"
 value={stats.activeUsers}
 icon={<TrendingUp size={20} />}
 />
 <StatCard
 label="Avg Processing Time"
 value={stats.avgProcessingTime}
 icon={<Clock size={20} />}
 />
 <StatCard
 label="Storage Used"
 value={stats.storageUsed}
 icon={<Film size={20} />}
 />
 </div>

 {/* Main content grid */}
 <div className="grid grid-cols-3 gap-6">
 <div className="col-span-2">
 <RecentJobsTable jobs={recentJobs} />
 </div>
 <div>
 <SystemAlerts alerts={alerts} />
 </div>
 </div>
 </div>
 );
}

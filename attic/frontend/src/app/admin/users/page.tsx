'use client';

import { useState, useEffect } from 'react';
import {
 Search,
 Loader,
 MoreHorizontal,
 ChevronLeft,
 ChevronRight,
 User,
 Shield,
 Crown,
 Mail,
 Ban,
 Edit,
 Trash2,
 CheckCircle,
 XCircle,
} from 'lucide-react';
import { cn, formatDate, formatRelativeTime } from '@/lib/utils';

// Role badge component
function RoleBadge({ role }: { role: string }) {
 const config: Record<string, { icon: React.ReactNode; label: string; className: string }> = {
 user: {
 icon: <User size={12} />,
 label: 'User',
 className: 'bg-baseline-100 text-baseline-600',
 },
 admin: {
 icon: <Shield size={12} />,
 label: 'Admin',
 className: 'bg-blue-100 text-blue-700',
 },
 super_admin: {
 icon: <Crown size={12} />,
 label: 'Super Admin',
 className: 'bg-purple-100 text-purple-700',
 },
 };

 const { icon, label, className } = config[role] || config.user;

 return (
 <span className={cn('inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium', className)}>
 {icon}
 {label}
 </span>
 );
}

// Plan badge component
function PlanBadge({ plan }: { plan: string }) {
 const config: Record<string, { label: string; className: string }> = {
 free: {
 label: 'Free',
 className: 'bg-baseline-100 text-baseline-600',
 },
 pro: {
 label: 'Pro',
 className: 'bg-court-100 text-court-700',
 },
 enterprise: {
 label: 'Enterprise',
 className: 'bg-purple-100 text-purple-700',
 },
 };

 const { label, className } = config[plan] || config.free;

 return (
 <span className={cn('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', className)}>
 {label}
 </span>
 );
}

// User action menu
function UserActionsMenu({ userId, isActive, onAction }: { userId: string; isActive: boolean; onAction: (action: string, id: string) => void }) {
 const [isOpen, setIsOpen] = useState(false);

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
 <button
 onClick={() => { onAction('edit', userId); setIsOpen(false); }}
 className="w-full flex items-center gap-2 px-3 py-2 text-sm text-baseline-600 hover:bg-baseline-50 transition-colors"
 >
 <Edit size={14} />
 Edit User
 </button>
 <button
 onClick={() => { onAction('email', userId); setIsOpen(false); }}
 className="w-full flex items-center gap-2 px-3 py-2 text-sm text-baseline-600 hover:bg-baseline-50 transition-colors"
 >
 <Mail size={14} />
 Send Email
 </button>
 <button
 onClick={() => { onAction(isActive ? 'suspend' : 'activate', userId); setIsOpen(false); }}
 className="w-full flex items-center gap-2 px-3 py-2 text-sm text-baseline-600 hover:bg-baseline-50 transition-colors"
 >
 <Ban size={14} />
 {isActive ? 'Suspend' : 'Activate'}
 </button>
 <button
 onClick={() => { onAction('delete', userId); setIsOpen(false); }}
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
interface AdminUser {
 id: string;
 email: string;
 fullName?: string;
 role: string;
 plan: string;
 isActive: boolean;
 jobsCount: number;
 usageMinutesRemaining: number;
 createdAt: string;
 lastLoginAt?: string;
}

const MOCK_USERS: AdminUser[] = [
 { id: 'user-001', email: 'john@example.com', fullName: 'John Doe', role: 'user', plan: 'pro', isActive: true, jobsCount: 45, usageMinutesRemaining: 120, createdAt: new Date(Date.now() - 30 * 86400000).toISOString(), lastLoginAt: new Date(Date.now() - 3600000).toISOString() },
 { id: 'user-002', email: 'jane@example.com', fullName: 'Jane Smith', role: 'admin', plan: 'enterprise', isActive: true, jobsCount: 128, usageMinutesRemaining: 999, createdAt: new Date(Date.now() - 90 * 86400000).toISOString(), lastLoginAt: new Date(Date.now() - 1800000).toISOString() },
 { id: 'user-003', email: 'mike@example.com', role: 'user', plan: 'free', isActive: true, jobsCount: 3, usageMinutesRemaining: 5, createdAt: new Date(Date.now() - 7 * 86400000).toISOString(), lastLoginAt: new Date(Date.now() - 86400000).toISOString() },
 { id: 'user-004', email: 'sarah@example.com', fullName: 'Sarah Wilson', role: 'user', plan: 'pro', isActive: false, jobsCount: 22, usageMinutesRemaining: 0, createdAt: new Date(Date.now() - 60 * 86400000).toISOString() },
 { id: 'user-005', email: 'alex@example.com', fullName: 'Alex Chen', role: 'user', plan: 'free', isActive: true, jobsCount: 8, usageMinutesRemaining: 2, createdAt: new Date(Date.now() - 14 * 86400000).toISOString(), lastLoginAt: new Date(Date.now() - 7200000).toISOString() },
 { id: 'user-006', email: 'admin@swishvision.com', fullName: 'System Admin', role: 'super_admin', plan: 'enterprise', isActive: true, jobsCount: 0, usageMinutesRemaining: 999, createdAt: new Date(Date.now() - 365 * 86400000).toISOString(), lastLoginAt: new Date(Date.now() - 300000).toISOString() },
 { id: 'user-007', email: 'chris@example.com', fullName: 'Chris Johnson', role: 'user', plan: 'pro', isActive: true, jobsCount: 67, usageMinutesRemaining: 85, createdAt: new Date(Date.now() - 45 * 86400000).toISOString(), lastLoginAt: new Date(Date.now() - 43200000).toISOString() },
 { id: 'user-008', email: 'emma@example.com', role: 'user', plan: 'free', isActive: true, jobsCount: 1, usageMinutesRemaining: 10, createdAt: new Date(Date.now() - 2 * 86400000).toISOString(), lastLoginAt: new Date(Date.now() - 86400000).toISOString() },
];

const ROLE_FILTERS = [
 { label: 'All Roles', value: '' },
 { label: 'Users', value: 'user' },
 { label: 'Admins', value: 'admin' },
 { label: 'Super Admins', value: 'super_admin' },
];

const PLAN_FILTERS = [
 { label: 'All Plans', value: '' },
 { label: 'Free', value: 'free' },
 { label: 'Pro', value: 'pro' },
 { label: 'Enterprise', value: 'enterprise' },
];

export default function AdminUsersPage() {
 const [users, setUsers] = useState<AdminUser[]>([]);
 const [isLoading, setIsLoading] = useState(true);
 const [searchQuery, setSearchQuery] = useState('');
 const [roleFilter, setRoleFilter] = useState('');
 const [planFilter, setPlanFilter] = useState('');
 const [page, setPage] = useState(1);
 const [selectedUsers, setSelectedUsers] = useState<Set<string>>(new Set());

 const perPage = 10;
 const totalPages = Math.ceil(MOCK_USERS.length / perPage);

 // Simulate loading
 useEffect(() => {
 const timer = setTimeout(() => {
 setUsers(MOCK_USERS);
 setIsLoading(false);
 }, 500);
 return () => clearTimeout(timer);
 }, []);

 // Filter users
 const filteredUsers = users.filter((user) => {
 if (roleFilter && user.role !== roleFilter) return false;
 if (planFilter && user.plan !== planFilter) return false;
 if (searchQuery) {
 const query = searchQuery.toLowerCase();
 return (
 user.email.toLowerCase().includes(query) ||
 (user.fullName && user.fullName.toLowerCase().includes(query))
 );
 }
 return true;
 });

 const handleUserAction = (action: string, userId: string) => {
 console.log(`Action: ${action} on user: ${userId}`);
 // In production, this would call the API
 };

 const toggleUserSelection = (userId: string) => {
 const newSelection = new Set(selectedUsers);
 if (newSelection.has(userId)) {
 newSelection.delete(userId);
 } else {
 newSelection.add(userId);
 }
 setSelectedUsers(newSelection);
 };

 const toggleAllSelection = () => {
 if (selectedUsers.size === filteredUsers.length) {
 setSelectedUsers(new Set());
 } else {
 setSelectedUsers(new Set(filteredUsers.map((u) => u.id)));
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
 <div className="flex items-center justify-between mb-6">
 <div>
 <h1 className="text-2xl font-bold text-blacktop">Users</h1>
 <p className="text-baseline-500 mt-1">Manage user accounts and permissions</p>
 </div>
 <button className="btn-primary py-2.5 px-5">
 Add User
 </button>
 </div>

 {/* Stats cards */}
 <div className="grid grid-cols-4 gap-4 mb-6">
 <div className="bg-white border border-baseline-100 rounded-lg p-4">
 <p className="text-sm text-baseline-500">Total Users</p>
 <p className="text-2xl font-bold text-blacktop mt-1">{users.length}</p>
 </div>
 <div className="bg-white border border-baseline-100 rounded-lg p-4">
 <p className="text-sm text-baseline-500">Active Users</p>
 <p className="text-2xl font-bold text-blacktop mt-1">{users.filter((u) => u.isActive).length}</p>
 </div>
 <div className="bg-white border border-baseline-100 rounded-lg p-4">
 <p className="text-sm text-baseline-500">Pro Users</p>
 <p className="text-2xl font-bold text-blacktop mt-1">{users.filter((u) => u.plan === 'pro').length}</p>
 </div>
 <div className="bg-white border border-baseline-100 rounded-lg p-4">
 <p className="text-sm text-baseline-500">Enterprise Users</p>
 <p className="text-2xl font-bold text-blacktop mt-1">{users.filter((u) => u.plan === 'enterprise').length}</p>
 </div>
 </div>

 {/* Filters bar */}
 <div className="flex items-center justify-between gap-4 mb-6">
 <div className="flex items-center gap-3">
 {/* Search */}
 <div className="relative">
 <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-baseline-400" />
 <input
 type="text"
 placeholder="Search users..."
 value={searchQuery}
 onChange={(e) => setSearchQuery(e.target.value)}
 className="w-64 pl-9 pr-4 py-2 rounded-lg border border-baseline-200 text-sm focus:outline-none focus:border-court"
 />
 </div>

 {/* Role filter */}
 <select
 value={roleFilter}
 onChange={(e) => setRoleFilter(e.target.value)}
 className="px-3 py-2 rounded-lg border border-baseline-200 text-sm focus:outline-none focus:border-court bg-white"
 >
 {ROLE_FILTERS.map((filter) => (
 <option key={filter.value} value={filter.value}>
 {filter.label}
 </option>
 ))}
 </select>

 {/* Plan filter */}
 <select
 value={planFilter}
 onChange={(e) => setPlanFilter(e.target.value)}
 className="px-3 py-2 rounded-lg border border-baseline-200 text-sm focus:outline-none focus:border-court bg-white"
 >
 {PLAN_FILTERS.map((filter) => (
 <option key={filter.value} value={filter.value}>
 {filter.label}
 </option>
 ))}
 </select>
 </div>

 {/* Bulk actions */}
 {selectedUsers.size > 0 && (
 <div className="flex items-center gap-2">
 <span className="text-sm text-baseline-500">{selectedUsers.size} selected</span>
 <button className="btn-secondary text-sm py-1.5 px-3">
 Send Email
 </button>
 <button className="btn-secondary text-sm py-1.5 px-3 text-red-600 border-red-200 hover:bg-red-50">
 Suspend Selected
 </button>
 </div>
 )}
 </div>

 {/* Users table */}
 <div className="bg-white border border-baseline-100 rounded-xl overflow-hidden">
 <table className="w-full">
 <thead>
 <tr className="bg-baseline-50 text-left text-xs text-baseline-500 uppercase tracking-wider">
 <th className="px-4 py-3 w-8">
 <input
 type="checkbox"
 checked={selectedUsers.size === filteredUsers.length && filteredUsers.length > 0}
 onChange={toggleAllSelection}
 className="rounded border-baseline-300"
 />
 </th>
 <th className="px-4 py-3">User</th>
 <th className="px-4 py-3">Role</th>
 <th className="px-4 py-3">Plan</th>
 <th className="px-4 py-3">Status</th>
 <th className="px-4 py-3">Jobs</th>
 <th className="px-4 py-3">Last Login</th>
 <th className="px-4 py-3 w-12"></th>
 </tr>
 </thead>
 <tbody className="divide-y divide-baseline-100">
 {filteredUsers.map((user) => (
 <tr key={user.id} className={cn('hover:bg-baseline-50 transition-colors', selectedUsers.has(user.id) && 'bg-court-50')}>
 <td className="px-4 py-3">
 <input
 type="checkbox"
 checked={selectedUsers.has(user.id)}
 onChange={() => toggleUserSelection(user.id)}
 className="rounded border-baseline-300"
 />
 </td>
 <td className="px-4 py-3">
 <div className="flex items-center gap-3">
 <div className="w-8 h-8 bg-court rounded-full flex items-center justify-center text-white text-sm font-medium">
 {(user.fullName || user.email).charAt(0).toUpperCase()}
 </div>
 <div>
 <p className="font-medium text-blacktop">{user.fullName || user.email.split('@')[0]}</p>
 <p className="text-xs text-baseline-500">{user.email}</p>
 </div>
 </div>
 </td>
 <td className="px-4 py-3">
 <RoleBadge role={user.role} />
 </td>
 <td className="px-4 py-3">
 <PlanBadge plan={user.plan} />
 </td>
 <td className="px-4 py-3">
 <span className={cn('inline-flex items-center gap-1 text-sm', user.isActive ? 'text-green-600' : 'text-red-600')}>
 {user.isActive ? <CheckCircle size={14} /> : <XCircle size={14} />}
 {user.isActive ? 'Active' : 'Suspended'}
 </span>
 </td>
 <td className="px-4 py-3 text-sm text-baseline-600">
 {user.jobsCount}
 </td>
 <td className="px-4 py-3 text-sm text-baseline-600">
 {user.lastLoginAt ? formatRelativeTime(user.lastLoginAt) : 'Never'}
 </td>
 <td className="px-4 py-3">
 <UserActionsMenu userId={user.id} isActive={user.isActive} onAction={handleUserAction} />
 </td>
 </tr>
 ))}
 </tbody>
 </table>

 {filteredUsers.length === 0 && (
 <div className="text-center py-12 text-baseline-500">
 No users found matching your criteria
 </div>
 )}
 </div>

 {/* Pagination */}
 {totalPages > 1 && (
 <div className="flex items-center justify-between mt-4">
 <p className="text-sm text-baseline-500">
 Showing {(page - 1) * perPage + 1} to {Math.min(page * perPage, filteredUsers.length)} of {filteredUsers.length} users
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

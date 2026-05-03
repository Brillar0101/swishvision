'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, Film, Users, Settings, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';

// Navigation items
const NAV_ITEMS = [
 { href: '/admin', label: 'Dashboard', icon: LayoutDashboard },
 { href: '/admin/jobs', label: 'Jobs', icon: Film },
 { href: '/admin/users', label: 'Users', icon: Users },
 { href: '/admin/settings', label: 'Settings', icon: Settings },
];

export default function AdminLayout({ children }: { children: React.ReactNode }) {
 const pathname = usePathname();

 return (
 <div className="min-h-screen bg-baseline-50">
 {/* Top navbar */}
 <header className="h-16 bg-blacktop flex items-center justify-between px-6">
 <Link href="/admin" className="flex items-center gap-2">
 <span className="text-xl font-bold text-white">
 SWISH<span className="text-court">/</span>VISION
 </span>
 <span className="ml-2 px-2 py-0.5 bg-court/20 text-court text-xs font-medium rounded-full">
 ADMIN
 </span>
 </Link>
 <div className="flex items-center gap-4">
 <Link
 href="/"
 className="text-sm text-white/70 hover:text-white transition-colors"
 >
 View Site
 </Link>
 <div className="w-8 h-8 bg-court rounded-full flex items-center justify-center text-white text-sm font-medium">
 A
 </div>
 </div>
 </header>

 <div className="flex">
 {/* Sidebar */}
 <aside className="w-64 min-h-[calc(100vh-4rem)] bg-white border-r border-baseline-100">
 <nav className="p-4 space-y-1">
 {NAV_ITEMS.map((item) => {
 const isActive = pathname === item.href || (item.href !== '/admin' && pathname.startsWith(item.href));
 return (
 <Link
 key={item.href}
 href={item.href}
 className={cn(
 'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
 isActive
 ? 'bg-court text-white'
 : 'text-baseline-600 hover:bg-baseline-50 hover:text-blacktop'
 )}
 >
 <item.icon size={18} />
 {item.label}
 </Link>
 );
 })}
 </nav>
 </aside>

 {/* Main content */}
 <main className="flex-1 p-8">{children}</main>
 </div>
 </div>
 );
}

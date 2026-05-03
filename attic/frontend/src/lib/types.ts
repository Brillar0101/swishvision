/**
 * Common types used throughout the application.
 */

// Re-export API types
export type {
  User,
  Team,
  TeamConfig,
  ProcessingConfig,
  VisualizationConfig,
  DetectedPlayer,
  JobVideoUrls,
  JobSummary,
  JobDetail,
  JobListResponse,
  AuthResponse,
} from './api';

// Job status enum
export type JobStatus =
  | 'pending'
  | 'uploading'
  | 'queued'
  | 'processing'
  | 'rendering'
  | 'completed'
  | 'failed'
  | 'cancelled';

// User role enum
export type UserRole = 'user' | 'admin' | 'super_admin';

// User plan enum
export type UserPlan = 'free' | 'pro' | 'enterprise';

// Form state for creating a job
export interface JobFormState {
  video: File | null;
  teamA: {
    name: string;
    color: string;
    roster: Record<string, string>;
  };
  teamB: {
    name: string;
    color: string;
    roster: Record<string, string>;
  };
  duration: 'full' | 'custom';
  maxDurationSeconds: number;
  options: {
    enableSegmentation: boolean;
    enableJerseyDetection: boolean;
    enableTacticalView: boolean;
  };
}

// Color preset for teams
export interface ColorPreset {
  name: string;
  hex: string;
}

// Default color presets
export const COLOR_PRESETS: ColorPreset[] = [
  { name: 'Red', hex: '#DC2626' },
  { name: 'Orange', hex: '#EA580C' },
  { name: 'Yellow', hex: '#CA8A04' },
  { name: 'Green', hex: '#16A34A' },
  { name: 'Blue', hex: '#2563EB' },
  { name: 'Purple', hex: '#7C3AED' },
  { name: 'Pink', hex: '#DB2777' },
  { name: 'Navy', hex: '#1E3A5F' },
  { name: 'Black', hex: '#171717' },
  { name: 'White', hex: '#FFFFFF' },
];

// Navigation item
export interface NavItem {
  label: string;
  href: string;
  icon?: string;
  adminOnly?: boolean;
}

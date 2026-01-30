'use client';

import { useState } from 'react';
import { Save, RefreshCw, AlertTriangle, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

// Settings section component
interface SettingsSectionProps {
 title: string;
 description: string;
 children: React.ReactNode;
}

function SettingsSection({ title, description, children }: SettingsSectionProps) {
 return (
 <div className="bg-white border border-baseline-100 rounded-xl p-6">
 <div className="mb-6">
 <h3 className="text-lg font-medium text-blacktop">{title}</h3>
 <p className="text-sm text-baseline-500 mt-1">{description}</p>
 </div>
 {children}
 </div>
 );
}

// Toggle switch component
interface ToggleSwitchProps {
 label: string;
 description?: string;
 checked: boolean;
 onChange: (checked: boolean) => void;
}

function ToggleSwitch({ label, description, checked, onChange }: ToggleSwitchProps) {
 return (
 <div className="flex items-center justify-between py-3">
 <div>
 <p className="text-sm font-medium text-blacktop">{label}</p>
 {description && <p className="text-xs text-baseline-500 mt-0.5">{description}</p>}
 </div>
 <button
 onClick={() => onChange(!checked)}
 className={cn(
 'relative w-11 h-6 rounded-full transition-colors',
 checked ? 'bg-court' : 'bg-baseline-200'
 )}
 >
 <span
 className={cn(
 'absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform',
 checked && 'translate-x-5'
 )}
 />
 </button>
 </div>
 );
}

// Input field component
interface InputFieldProps {
 label: string;
 description?: string;
 type?: string;
 value: string | number;
 onChange: (value: string) => void;
 placeholder?: string;
 suffix?: string;
}

function InputField({ label, description, type = 'text', value, onChange, placeholder, suffix }: InputFieldProps) {
 return (
 <div className="py-3">
 <label className="block text-sm font-medium text-blacktop mb-1">{label}</label>
 {description && <p className="text-xs text-baseline-500 mb-2">{description}</p>}
 <div className="flex items-center gap-2">
 <input
 type={type}
 value={value}
 onChange={(e) => onChange(e.target.value)}
 placeholder={placeholder}
 className="flex-1 max-w-xs px-3 py-2 rounded-lg border border-baseline-200 text-sm focus:outline-none focus:border-court"
 />
 {suffix && <span className="text-sm text-baseline-500">{suffix}</span>}
 </div>
 </div>
 );
}

// Select field component
interface SelectFieldProps {
 label: string;
 description?: string;
 value: string;
 onChange: (value: string) => void;
 options: { label: string; value: string }[];
}

function SelectField({ label, description, value, onChange, options }: SelectFieldProps) {
 return (
 <div className="py-3">
 <label className="block text-sm font-medium text-blacktop mb-1">{label}</label>
 {description && <p className="text-xs text-baseline-500 mb-2">{description}</p>}
 <select
 value={value}
 onChange={(e) => onChange(e.target.value)}
 className="w-full max-w-xs px-3 py-2 border border-baseline-200 text-sm focus:outline-none focus:border-court bg-white"
 >
 {options.map((option) => (
 <option key={option.value} value={option.value}>
 {option.label}
 </option>
 ))}
 </select>
 </div>
 );
}

export default function AdminSettingsPage() {
 const [isSaving, setIsSaving] = useState(false);
 const [saveSuccess, setSaveSuccess] = useState(false);

 // Processing settings
 const [processingSettings, setProcessingSettings] = useState({
 maxConcurrentJobs: '3',
 defaultMaxDuration: '300',
 gpuMemoryLimit: '24',
 enableAutoRetry: true,
 maxRetryAttempts: '3',
 });

 // Feature flags
 const [featureFlags, setFeatureFlags] = useState({
 enableSegmentation: true,
 enableJerseyDetection: true,
 enableTacticalView: true,
 enableNewUserSignup: true,
 maintenanceMode: false,
 });

 // Storage settings
 const [storageSettings, setStorageSettings] = useState({
 s3Bucket: 'swishvision-uploads',
 s3Region: 'us-west-2',
 videoRetentionDays: '30',
 maxUploadSizeMB: '500',
 });

 // Email settings
 const [emailSettings, setEmailSettings] = useState({
 fromEmail: 'noreply@swishvision.com',
 sendJobComplete: true,
 sendJobFailed: true,
 sendWeeklySummary: false,
 });

 const handleSave = async () => {
 setIsSaving(true);
 setSaveSuccess(false);
 // Simulate API call
 await new Promise((resolve) => setTimeout(resolve, 1000));
 setIsSaving(false);
 setSaveSuccess(true);
 setTimeout(() => setSaveSuccess(false), 3000);
 };

 return (
 <div>
 <div className="flex items-center justify-between mb-6">
 <div>
 <h1 className="text-2xl font-bold text-blacktop">Settings</h1>
 <p className="text-baseline-500 mt-1">Configure system settings and preferences</p>
 </div>
 <div className="flex items-center gap-3">
 {saveSuccess && (
 <span className="flex items-center gap-2 text-sm text-green-600">
 <CheckCircle size={16} />
 Settings saved
 </span>
 )}
 <button
 onClick={handleSave}
 disabled={isSaving}
 className="btn-primary py-2.5 px-5 flex items-center gap-2"
 >
 {isSaving ? (
 <>
 <RefreshCw size={16} className="animate-spin" />
 Saving...
 </>
 ) : (
 <>
 <Save size={16} />
 Save Changes
 </>
 )}
 </button>
 </div>
 </div>

 <div className="space-y-6">
 {/* Processing Settings */}
 <SettingsSection
 title="Processing"
 description="Configure video processing behavior and limits"
 >
 <div className="divide-y divide-baseline-100">
 <InputField
 label="Max Concurrent Jobs"
 description="Maximum number of jobs that can process simultaneously"
 type="number"
 value={processingSettings.maxConcurrentJobs}
 onChange={(v) => setProcessingSettings({ ...processingSettings, maxConcurrentJobs: v })}
 />
 <InputField
 label="Default Max Duration"
 description="Default maximum video duration for processing"
 type="number"
 value={processingSettings.defaultMaxDuration}
 onChange={(v) => setProcessingSettings({ ...processingSettings, defaultMaxDuration: v })}
 suffix="seconds"
 />
 <InputField
 label="GPU Memory Limit"
 description="Maximum GPU memory allocation per job"
 type="number"
 value={processingSettings.gpuMemoryLimit}
 onChange={(v) => setProcessingSettings({ ...processingSettings, gpuMemoryLimit: v })}
 suffix="GB"
 />
 <ToggleSwitch
 label="Enable Auto Retry"
 description="Automatically retry failed jobs"
 checked={processingSettings.enableAutoRetry}
 onChange={(v) => setProcessingSettings({ ...processingSettings, enableAutoRetry: v })}
 />
 {processingSettings.enableAutoRetry && (
 <InputField
 label="Max Retry Attempts"
 type="number"
 value={processingSettings.maxRetryAttempts}
 onChange={(v) => setProcessingSettings({ ...processingSettings, maxRetryAttempts: v })}
 />
 )}
 </div>
 </SettingsSection>

 {/* Feature Flags */}
 <SettingsSection
 title="Feature Flags"
 description="Enable or disable specific features"
 >
 <div className="divide-y divide-baseline-100">
 <ToggleSwitch
 label="Player Segmentation"
 description="Enable player outline masks in output videos"
 checked={featureFlags.enableSegmentation}
 onChange={(v) => setFeatureFlags({ ...featureFlags, enableSegmentation: v })}
 />
 <ToggleSwitch
 label="Jersey Detection"
 description="Enable jersey number OCR and player identification"
 checked={featureFlags.enableJerseyDetection}
 onChange={(v) => setFeatureFlags({ ...featureFlags, enableJerseyDetection: v })}
 />
 <ToggleSwitch
 label="Tactical View"
 description="Enable court overview minimap in output videos"
 checked={featureFlags.enableTacticalView}
 onChange={(v) => setFeatureFlags({ ...featureFlags, enableTacticalView: v })}
 />
 <ToggleSwitch
 label="New User Signup"
 description="Allow new users to register accounts"
 checked={featureFlags.enableNewUserSignup}
 onChange={(v) => setFeatureFlags({ ...featureFlags, enableNewUserSignup: v })}
 />
 <div className="py-3">
 <div className="flex items-center justify-between">
 <div>
 <p className="text-sm font-medium text-blacktop flex items-center gap-2">
 Maintenance Mode
 {featureFlags.maintenanceMode && (
 <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs rounded-full">Active</span>
 )}
 </p>
 <p className="text-xs text-baseline-500 mt-0.5">Disable all processing and show maintenance message</p>
 </div>
 <button
 onClick={() => setFeatureFlags({ ...featureFlags, maintenanceMode: !featureFlags.maintenanceMode })}
 className={cn(
 'relative w-11 h-6 rounded-full transition-colors',
 featureFlags.maintenanceMode ? 'bg-red-500' : 'bg-baseline-200'
 )}
 >
 <span
 className={cn(
 'absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform',
 featureFlags.maintenanceMode && 'translate-x-5'
 )}
 />
 </button>
 </div>
 {featureFlags.maintenanceMode && (
 <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
 <AlertTriangle size={16} className="text-red-600 flex-shrink-0 mt-0.5" />
 <p className="text-sm text-red-700">
 Maintenance mode is active. Users cannot submit new jobs and will see a maintenance message.
 </p>
 </div>
 )}
 </div>
 </div>
 </SettingsSection>

 {/* Storage Settings */}
 <SettingsSection
 title="Storage"
 description="Configure S3 storage settings"
 >
 <div className="divide-y divide-baseline-100">
 <InputField
 label="S3 Bucket"
 value={storageSettings.s3Bucket}
 onChange={(v) => setStorageSettings({ ...storageSettings, s3Bucket: v })}
 />
 <SelectField
 label="S3 Region"
 value={storageSettings.s3Region}
 onChange={(v) => setStorageSettings({ ...storageSettings, s3Region: v })}
 options={[
 { label: 'US West 2 (Oregon)', value: 'us-west-2' },
 { label: 'US East 1 (N. Virginia)', value: 'us-east-1' },
 { label: 'EU West 1 (Ireland)', value: 'eu-west-1' },
 { label: 'AP Southeast 1 (Singapore)', value: 'ap-southeast-1' },
 ]}
 />
 <InputField
 label="Video Retention"
 description="How long to keep processed videos"
 type="number"
 value={storageSettings.videoRetentionDays}
 onChange={(v) => setStorageSettings({ ...storageSettings, videoRetentionDays: v })}
 suffix="days"
 />
 <InputField
 label="Max Upload Size"
 description="Maximum allowed file upload size"
 type="number"
 value={storageSettings.maxUploadSizeMB}
 onChange={(v) => setStorageSettings({ ...storageSettings, maxUploadSizeMB: v })}
 suffix="MB"
 />
 </div>
 </SettingsSection>

 {/* Email Settings */}
 <SettingsSection
 title="Email Notifications"
 description="Configure email notification settings"
 >
 <div className="divide-y divide-baseline-100">
 <InputField
 label="From Email"
 type="email"
 value={emailSettings.fromEmail}
 onChange={(v) => setEmailSettings({ ...emailSettings, fromEmail: v })}
 />
 <ToggleSwitch
 label="Job Complete Notifications"
 description="Send email when a job completes successfully"
 checked={emailSettings.sendJobComplete}
 onChange={(v) => setEmailSettings({ ...emailSettings, sendJobComplete: v })}
 />
 <ToggleSwitch
 label="Job Failed Notifications"
 description="Send email when a job fails"
 checked={emailSettings.sendJobFailed}
 onChange={(v) => setEmailSettings({ ...emailSettings, sendJobFailed: v })}
 />
 <ToggleSwitch
 label="Weekly Summary"
 description="Send weekly usage summary to users"
 checked={emailSettings.sendWeeklySummary}
 onChange={(v) => setEmailSettings({ ...emailSettings, sendWeeklySummary: v })}
 />
 </div>
 </SettingsSection>
 </div>
 </div>
 );
}

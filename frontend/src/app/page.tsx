'use client'

import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Upload, 
  Play, 
  Users, 
  Activity, 
  Target,
  ChevronRight,
  Check,
  Loader2,
  Film,
  BarChart3,
  Layers,
  Zap,
  RefreshCw,
  Download,
  X,
  Settings,
  Eye,
  Route,
  Shirt
} from 'lucide-react'

// API Configuration
const API_BASE = ''  // Uses Next.js rewrites to proxy to backend

interface JobStatus {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  message: string
  created_at: string
  completed_at?: string
  result?: AnalysisResult
}

interface AnalysisResult {
  video_url: string
  sample_frames: string[]
  stats: {
    frames_processed: number
    players_tracked: number
    team_counts: {
      team_0: number
      team_1: number
      referee: number
    }
    players: Array<{
      id: number
      team: number
      team_name: string
      class: string
    }>
  }
}

interface AnalysisOptions {
  playerTracking: boolean
  teamClassification: boolean
  tacticalView: boolean
  courtDetection: boolean
}

type AppState = 'upload' | 'processing' | 'results'

export default function Home() {
  const [appState, setAppState] = useState<AppState>('upload')
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showFilters, setShowFilters] = useState(false)
  const [headerVisible, setHeaderVisible] = useState(true)
  const [lastScrollY, setLastScrollY] = useState(0)
  
  // Analysis options
  const [analysisOptions, setAnalysisOptions] = useState<AnalysisOptions>({
    playerTracking: true,
    teamClassification: true,
    tacticalView: true,
    courtDetection: true,
  })

  // Handle scroll to show/hide header
  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY
      
      if (currentScrollY < 50) {
        setHeaderVisible(true)
      } else if (currentScrollY > lastScrollY) {
        setHeaderVisible(false)
      } else {
        setHeaderVisible(true)
      }
      
      setLastScrollY(currentScrollY)
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [lastScrollY])

  // Poll for job status
  useEffect(() => {
    if (!jobId || appState !== 'processing') return

    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status/${jobId}`)
        const data: JobStatus = await res.json()
        setJobStatus(data)

        if (data.status === 'completed') {
          setAppState('results')
          clearInterval(pollInterval)
        } else if (data.status === 'failed') {
          setError(data.message)
          setAppState('upload')
          clearInterval(pollInterval)
        }
      } catch (err) {
        console.error('Failed to poll status:', err)
      }
    }, 1000)

    return () => clearInterval(pollInterval)
  }, [jobId, appState])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file)
      setError(null)
    } else {
      setError('Please upload a video file')
    }
  }, [])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setError(null)
    }
  }, [])

  const toggleOption = (key: keyof AnalysisOptions) => {
    setAnalysisOptions(prev => ({
      ...prev,
      [key]: !prev[key]
    }))
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    try {
      setError(null)
      const formData = new FormData()
      formData.append('file', selectedFile)
      
      // Add analysis options to form data
      formData.append('options', JSON.stringify(analysisOptions))

      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        throw new Error('Upload failed')
      }

      const data = await res.json()
      setJobId(data.job_id)
      setAppState('processing')
    } catch (err) {
      setError('Failed to upload video. Please try again.')
      console.error(err)
    }
  }

  const handleReset = () => {
    setAppState('upload')
    setSelectedFile(null)
    setJobId(null)
    setJobStatus(null)
    setError(null)
  }

  const filterOptions = [
    { key: 'playerTracking' as const, icon: Users, label: 'Player Tracking', desc: 'Track player movements with SAM2' },
    { key: 'teamClassification' as const, icon: Shirt, label: 'Team Classification', desc: 'Identify teams by jersey colors' },
    { key: 'tacticalView' as const, icon: Route, label: 'Tactical View', desc: "Bird's eye court visualization" },
    { key: 'courtDetection' as const, icon: Eye, label: 'Court Detection', desc: 'Detect court boundaries & keypoints' },
  ]

  return (
    <main className="min-h-screen">
      {/* Header - Hides on scroll */}
      <motion.header 
        className="fixed top-0 left-0 right-0 z-50"
        initial={{ y: 0 }}
        animate={{ y: headerVisible ? 0 : -100 }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
      >
        <div className="glass mx-4 mt-4 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center">
              <Target className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-white">Swish Vision</h1>
              <p className="text-xs text-white/50">AI Basketball Analytics</p>
            </div>
          </div>
          
          {appState !== 'upload' && (
            <button
              onClick={handleReset}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-sm text-white/70 hover:text-white"
            >
              <RefreshCw className="w-4 h-4" />
              New Analysis
            </button>
          )}
        </div>
      </motion.header>

      <div className="pt-28 pb-12 px-4">
        <AnimatePresence mode="wait">
          {/* Upload State */}
          {appState === 'upload' && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-2xl mx-auto"
            >
              {/* Hero */}
              <div className="text-center mb-12">
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.1 }}
                  className="w-20 h-20 mx-auto mb-6 rounded-3xl bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center orange-glow"
                >
                  <Zap className="w-10 h-10 text-white" />
                </motion.div>
                <h2 className="text-4xl font-bold text-white mb-4">
                  Analyze Your Game
                </h2>
                <p className="text-lg text-white/60 max-w-md mx-auto">
                  Upload basketball footage and get instant AI-powered analysis with player tracking, team classification, and tactical views.
                </p>
              </div>

              {/* Upload Zone */}
              <div
                className={`upload-zone p-12 text-center cursor-pointer transition-all ${
                  isDragging ? 'drag-over' : ''
                } ${selectedFile ? 'orange-border' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input')?.click()}
              >
                <input
                  id="file-input"
                  type="file"
                  accept="video/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                
                {selectedFile ? (
                  <div className="space-y-4">
                    <div className="w-16 h-16 mx-auto rounded-2xl bg-orange-500/20 flex items-center justify-center">
                      <Film className="w-8 h-8 text-orange-500" />
                    </div>
                    <div>
                      <p className="text-white font-medium">{selectedFile.name}</p>
                      <p className="text-white/50 text-sm">
                        {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setSelectedFile(null)
                      }}
                      className="text-white/50 hover:text-white text-sm flex items-center gap-1 mx-auto"
                    >
                      <X className="w-4 h-4" />
                      Remove
                    </button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="w-16 h-16 mx-auto rounded-2xl bg-white/5 flex items-center justify-center">
                      <Upload className="w-8 h-8 text-white/50" />
                    </div>
                    <div>
                      <p className="text-white font-medium">
                        Drop your video here
                      </p>
                      <p className="text-white/50 text-sm">
                        or click to browse • MP4, MOV, AVI supported
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Analysis Options Toggle */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.15 }}
                className="mt-6"
              >
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="w-full flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/8 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <Settings className="w-5 h-5 text-orange-500" />
                    <span className="text-white font-medium">Analysis Options</span>
                  </div>
                  <motion.div
                    animate={{ rotate: showFilters ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <ChevronRight className="w-5 h-5 text-white/50 rotate-90" />
                  </motion.div>
                </button>

                <AnimatePresence>
                  {showFilters && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className="overflow-hidden"
                    >
                      <div className="grid grid-cols-2 gap-3 mt-3">
                        {filterOptions.map((option) => (
                          <button
                            key={option.key}
                            onClick={() => toggleOption(option.key)}
                            className={`p-4 rounded-xl text-left transition-all ${
                              analysisOptions[option.key]
                                ? 'bg-orange-500/20 border border-orange-500/50'
                                : 'bg-white/5 border border-transparent hover:bg-white/8'
                            }`}
                          >
                            <div className="flex items-start gap-3">
                              <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                                analysisOptions[option.key]
                                  ? 'bg-orange-500'
                                  : 'bg-white/10'
                              }`}>
                                <option.icon className={`w-5 h-5 ${
                                  analysisOptions[option.key]
                                    ? 'text-white'
                                    : 'text-white/50'
                                }`} />
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className={`font-medium text-sm ${
                                  analysisOptions[option.key]
                                    ? 'text-white'
                                    : 'text-white/70'
                                }`}>
                                  {option.label}
                                </p>
                                <p className="text-xs text-white/40 mt-0.5 truncate">
                                  {option.desc}
                                </p>
                              </div>
                              <div className={`w-5 h-5 rounded-full flex items-center justify-center ${
                                analysisOptions[option.key]
                                  ? 'bg-orange-500'
                                  : 'bg-white/10'
                              }`}>
                                {analysisOptions[option.key] && (
                                  <Check className="w-3 h-3 text-white" />
                                )}
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>

              {/* Error Message */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm text-center"
                >
                  {error}
                </motion.div>
              )}

              {/* Upload Button */}
              <motion.button
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                onClick={handleUpload}
                disabled={!selectedFile}
                className="glass-button w-full mt-6 flex items-center justify-center gap-2"
              >
                <Play className="w-5 h-5" />
                Start Analysis
              </motion.button>

              {/* Features */}
              <div className="grid grid-cols-3 gap-4 mt-12">
                {[
                  { icon: Users, label: 'Player Tracking', desc: 'SAM2 segmentation' },
                  { icon: Layers, label: 'Team Detection', desc: 'Jersey color AI' },
                  { icon: BarChart3, label: 'Tactical View', desc: "Bird's eye analysis" },
                ].map((feature, i) => (
                  <motion.div
                    key={feature.label}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + i * 0.1 }}
                    className="stat-card text-center"
                  >
                    <feature.icon className="w-6 h-6 text-orange-500 mx-auto mb-2" />
                    <p className="text-white text-sm font-medium">{feature.label}</p>
                    <p className="text-white/40 text-xs">{feature.desc}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Processing State */}
          {appState === 'processing' && (
            <motion.div
              key="processing"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-xl mx-auto text-center"
            >
              <div className="glass-card p-12">
                {/* Animated Logo */}
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                  className="w-20 h-20 mx-auto mb-8 rounded-3xl bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center orange-glow"
                >
                  <Loader2 className="w-10 h-10 text-white" />
                </motion.div>

                <h2 className="text-2xl font-bold text-white mb-2">
                  Analyzing Video
                </h2>
                <p className="text-white/60 mb-8">
                  {jobStatus?.message || 'Preparing analysis...'}
                </p>

                {/* Progress Bar */}
                <div className="progress-bar h-2 mb-4">
                  <motion.div
                    className="progress-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${jobStatus?.progress || 0}%` }}
                  />
                </div>
                <p className="text-white/40 text-sm">
                  {jobStatus?.progress || 0}% complete
                </p>

                {/* Processing Steps */}
                <div className="mt-8 space-y-3 text-left">
                  {[
                    { label: 'Loading ML models', threshold: 10 },
                    { label: 'Detecting players', threshold: 30 },
                    { label: 'Tracking movement', threshold: 50 },
                    { label: 'Classifying teams', threshold: 70 },
                    { label: 'Generating output', threshold: 90 },
                  ].map((step) => {
                    const progress = jobStatus?.progress || 0
                    const isComplete = progress >= step.threshold
                    const isActive = progress >= step.threshold - 20 && progress < step.threshold
                    
                    return (
                      <div
                        key={step.label}
                        className={`flex items-center gap-3 p-3 rounded-xl transition-colors ${
                          isComplete
                            ? 'bg-orange-500/10'
                            : isActive
                            ? 'bg-white/5'
                            : ''
                        }`}
                      >
                        <div
                          className={`w-6 h-6 rounded-full flex items-center justify-center ${
                            isComplete
                              ? 'bg-orange-500'
                              : isActive
                              ? 'bg-white/20'
                              : 'bg-white/5'
                          }`}
                        >
                          {isComplete ? (
                            <Check className="w-4 h-4 text-white" />
                          ) : isActive ? (
                            <Loader2 className="w-4 h-4 text-white animate-spin" />
                          ) : (
                            <div className="w-2 h-2 rounded-full bg-white/20" />
                          )}
                        </div>
                        <span
                          className={`text-sm ${
                            isComplete
                              ? 'text-white'
                              : isActive
                              ? 'text-white/70'
                              : 'text-white/30'
                          }`}
                        >
                          {step.label}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            </motion.div>
          )}

          {/* Results State */}
          {appState === 'results' && jobStatus?.result && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-6xl mx-auto"
            >
              {/* Success Header */}
              <div className="text-center mb-8">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', bounce: 0.5 }}
                  className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center"
                >
                  <Check className="w-8 h-8 text-white" />
                </motion.div>
                <h2 className="text-3xl font-bold text-white mb-2">
                  Analysis Complete
                </h2>
                <p className="text-white/60">
                  {jobStatus.result.stats.frames_processed} frames processed •{' '}
                  {jobStatus.result.stats.players_tracked} players tracked
                </p>
              </div>

              {/* Main Content Grid */}
              <div className="grid lg:grid-cols-3 gap-6">
                {/* Video Player */}
                <div className="lg:col-span-2">
                  <div className="glass-card p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Film className="w-5 h-5 text-orange-500" />
                      Analyzed Video
                    </h3>
                    <div className="video-container aspect-video">
                      <video
                        src={jobStatus.result.video_url}
                        controls
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <div className="mt-4 flex gap-3">
                      <a
                        href={jobStatus.result.video_url}
                        download
                        className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-white/70 hover:text-white text-sm"
                      >
                        <Download className="w-4 h-4" />
                        Download Video
                      </a>
                    </div>
                  </div>
                </div>

                {/* Stats Panel */}
                <div className="space-y-6">
                  {/* Team Stats */}
                  <div className="glass-card p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Users className="w-5 h-5 text-orange-500" />
                      Team Breakdown
                    </h3>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-3 rounded-xl bg-orange-500/10">
                        <span className="text-white/70">Team A</span>
                        <span className="text-2xl font-bold text-orange-500">
                          {jobStatus.result.stats.team_counts.team_0}
                        </span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-xl bg-blue-500/10">
                        <span className="text-white/70">Team B</span>
                        <span className="text-2xl font-bold text-blue-400">
                          {jobStatus.result.stats.team_counts.team_1}
                        </span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-xl bg-green-500/10">
                        <span className="text-white/70">Referees</span>
                        <span className="text-2xl font-bold text-green-400">
                          {jobStatus.result.stats.team_counts.referee}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Player List */}
                  <div className="glass-card p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Activity className="w-5 h-5 text-orange-500" />
                      Players Detected
                    </h3>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {jobStatus.result.stats.players.map((player) => (
                        <div
                          key={player.id}
                          className="flex items-center justify-between p-3 rounded-xl bg-white/5"
                        >
                          <div className="flex items-center gap-3">
                            <div
                              className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold ${
                                player.team === -1
                                  ? 'bg-green-500/20 text-green-400'
                                  : player.team === 0
                                  ? 'bg-orange-500/20 text-orange-400'
                                  : 'bg-blue-500/20 text-blue-400'
                              }`}
                            >
                              #{player.id}
                            </div>
                            <span className="text-white/70 text-sm">
                              {player.team_name}
                            </span>
                          </div>
                          <ChevronRight className="w-4 h-4 text-white/30" />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Sample Frames */}
              <div className="mt-6">
                <div className="glass-card p-6">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Layers className="w-5 h-5 text-orange-500" />
                    Key Frames
                  </h3>
                  <div className="grid grid-cols-3 gap-4">
                    {jobStatus.result.sample_frames.map((frame, i) => (
                      <motion.div
                        key={frame}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.1 }}
                        className="aspect-video rounded-xl overflow-hidden bg-black/20 cursor-pointer hover:ring-2 ring-orange-500 transition-all"
                      >
                        <img
                          src={frame}
                          alt={`Frame ${i + 1}`}
                          className="w-full h-full object-cover"
                        />
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  )
}
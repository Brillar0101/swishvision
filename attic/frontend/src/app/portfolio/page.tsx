'use client'

import { motion } from 'framer-motion'
import {
  Target,
  Users,
  Layers,
  BarChart3,
  Code2,
  Cpu,
  Database,
  Globe,
  Github,
  ExternalLink,
  ChevronRight,
  Zap,
  Eye,
  Route,
  Shirt,
  Video,
  Image as ImageIcon,
  FileCode,
  Server,
  Brain,
  Sparkles
} from 'lucide-react'
import Image from 'next/image'
import Link from 'next/link'

export default function Portfolio() {
  const techStack = {
    frontend: [
      { name: 'Next.js 14', desc: 'React framework with App Router' },
      { name: 'TypeScript', desc: 'Type-safe JavaScript' },
      { name: 'Tailwind CSS', desc: 'Utility-first styling' },
      { name: 'Framer Motion', desc: 'Animation library' },
    ],
    backend: [
      { name: 'FastAPI', desc: 'High-performance Python API' },
      { name: 'Python 3.11+', desc: 'Backend language' },
      { name: 'Uvicorn', desc: 'ASGI server' },
      { name: 'Pydantic', desc: 'Data validation' },
    ],
    ml: [
      { name: 'PyTorch', desc: 'Deep learning framework' },
      { name: 'SAM2', desc: 'Segment Anything Model 2' },
      { name: 'Roboflow', desc: 'Computer vision APIs' },
      { name: 'OpenCV', desc: 'Video processing' },
    ],
    models: [
      { name: 'YOLOv8', desc: 'Player detection' },
      { name: 'SigLIP', desc: 'Team classification embeddings' },
      { name: 'RF-DETR', desc: 'Object detection' },
    ]
  }

  const features = [
    {
      icon: Users,
      title: 'Player Detection & Tracking',
      description: 'Real-time player identification using YOLOv8 and frame-to-frame tracking with SAM2 segmentation for continuous player correlation.',
      image: '/portfolio/player_detection.jpg'
    },
    {
      icon: Shirt,
      title: 'Team Classification',
      description: 'Automatic team identification through jersey color analysis using SigLIP embeddings and K-means clustering with UMAP dimensionality reduction.',
      image: '/portfolio/team_classification.jpg'
    },
    {
      icon: Route,
      title: 'Tactical Bird\'s Eye View',
      description: 'Homography-based court mapping that transforms game footage into real-time tactical visualizations with player positions in NBA court coordinates.',
      image: '/portfolio/tactical_view.jpg'
    },
    {
      icon: Eye,
      title: 'Court Detection',
      description: '33-point court keypoint detection system that identifies court boundaries, paint area, three-point line, and center court for accurate spatial mapping.',
      image: '/portfolio/full_tracking.jpg'
    }
  ]

  const stats = [
    { value: '4,800+', label: 'Lines of Code' },
    { value: '5', label: 'ML Models' },
    { value: '6', label: 'API Endpoints' },
    { value: '21', label: 'Python Modules' },
  ]

  const challenges = [
    {
      title: 'Multi-Model Pipeline Orchestration',
      problem: 'Coordinating multiple ML models (YOLOv8, SAM2, SigLIP, RF-DETR, Court Detection) with varying inference times and dependencies.',
      solution: 'Implemented a staged processing pipeline with progress tracking, where each model\'s output feeds into the next while maintaining frame-level synchronization.'
    },
    {
      title: 'Real-time Homography Calculation',
      problem: 'Static homography matrices fail when camera angles shift during gameplay, causing tactical view drift.',
      solution: 'Implemented per-frame keypoint detection with dynamic ViewTransformer updates, recalculating the homography matrix for each frame based on visible court markers.'
    }
  ]

  return (
    <main className="min-h-screen bg-[#0a0a0f]">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-orange-500/10 via-transparent to-transparent" />
        <div className="absolute top-20 left-1/4 w-96 h-96 bg-orange-500/20 rounded-full blur-3xl" />
        <div className="absolute top-40 right-1/4 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl" />

        <div className="relative max-w-6xl mx-auto px-6 pt-20 pb-32">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-orange-500/10 border border-orange-500/20 text-orange-400 text-sm mb-8">
              <Sparkles className="w-4 h-4" />
              Personal Project • 2024
            </div>

            <div className="flex items-center justify-center gap-4 mb-6">
              <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center shadow-lg shadow-orange-500/30">
                <Target className="w-10 h-10 text-white" />
              </div>
            </div>

            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
              Swish Vision
            </h1>

            <p className="text-xl md:text-2xl text-white/60 max-w-3xl mx-auto mb-8">
              AI-powered basketball game analysis platform for college programs.
              Real-time player tracking, team classification, and tactical visualization
              using state-of-the-art computer vision models.
            </p>

            <div className="flex items-center justify-center gap-4">
              <Link
                href="/"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-orange-500 to-orange-600 text-white font-medium hover:shadow-lg hover:shadow-orange-500/30 transition-all"
              >
                <Zap className="w-5 h-5" />
                Try Live Demo
              </Link>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-white/5 border border-white/10 text-white font-medium hover:bg-white/10 transition-all"
              >
                <Github className="w-5 h-5" />
                View Source
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="border-y border-white/10 bg-white/[0.02]">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold text-orange-500 mb-1">
                  {stat.value}
                </div>
                <div className="text-white/50 text-sm">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Problem & Solution */}
      <section className="py-24">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid md:grid-cols-2 gap-12"
          >
            <div className="p-8 rounded-3xl bg-red-500/5 border border-red-500/10">
              <div className="w-12 h-12 rounded-2xl bg-red-500/10 flex items-center justify-center mb-6">
                <Target className="w-6 h-6 text-red-400" />
              </div>
              <h2 className="text-2xl font-bold text-white mb-4">The Problem</h2>
              <p className="text-white/60 leading-relaxed">
                College basketball programs spend countless hours manually reviewing game footage
                to analyze player performance, team formations, and tactical patterns. Traditional
                video analysis is time-consuming, subjective, and lacks real-time insights. Coaches
                need automated tools that can quickly identify players, track movements, and provide
                bird's-eye tactical views without expensive enterprise solutions.
              </p>
            </div>

            <div className="p-8 rounded-3xl bg-green-500/5 border border-green-500/10">
              <div className="w-12 h-12 rounded-2xl bg-green-500/10 flex items-center justify-center mb-6">
                <Zap className="w-6 h-6 text-green-400" />
              </div>
              <h2 className="text-2xl font-bold text-white mb-4">The Solution</h2>
              <p className="text-white/60 leading-relaxed">
                SwishVision automates basketball game analysis using a sophisticated ML pipeline.
                Upload game footage and receive instant player detection, automatic team classification
                based on jersey colors, continuous player tracking across frames,
                and tactical bird's-eye court visualizations—all powered by state-of-the-art models like
                SAM2, YOLOv8, and SigLIP.
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features with Images */}
      <section className="py-24 bg-white/[0.01]">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">Core Features</h2>
            <p className="text-white/60 max-w-2xl mx-auto">
              Advanced computer vision capabilities powered by multiple ML models working in concert
            </p>
          </motion.div>

          <div className="space-y-24">
            {features.map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
                className={`grid md:grid-cols-2 gap-12 items-center ${
                  i % 2 === 1 ? 'md:flex-row-reverse' : ''
                }`}
              >
                <div className={i % 2 === 1 ? 'md:order-2' : ''}>
                  <div className="w-14 h-14 rounded-2xl bg-orange-500/10 flex items-center justify-center mb-6">
                    <feature.icon className="w-7 h-7 text-orange-500" />
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-4">{feature.title}</h3>
                  <p className="text-white/60 leading-relaxed text-lg">{feature.description}</p>
                </div>
                <div className={`relative ${i % 2 === 1 ? 'md:order-1' : ''}`}>
                  <div className="aspect-video rounded-2xl overflow-hidden border border-white/10 shadow-2xl shadow-black/50">
                    <img
                      src={feature.image}
                      alt={feature.title}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="absolute -inset-4 bg-gradient-to-r from-orange-500/20 to-blue-500/20 rounded-3xl blur-2xl -z-10" />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-24">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">Tech Stack</h2>
            <p className="text-white/60 max-w-2xl mx-auto">
              Modern technologies chosen for performance, developer experience, and ML capabilities
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { title: 'Frontend', icon: Globe, items: techStack.frontend, color: 'blue' },
              { title: 'Backend', icon: Server, items: techStack.backend, color: 'green' },
              { title: 'ML/CV', icon: Brain, items: techStack.ml, color: 'purple' },
              { title: 'Models', icon: Cpu, items: techStack.models, color: 'orange' },
            ].map((category, i) => (
              <motion.div
                key={category.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="p-6 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-white/10 transition-colors"
              >
                <div className={`w-12 h-12 rounded-xl bg-${category.color}-500/10 flex items-center justify-center mb-4`}>
                  <category.icon className={`w-6 h-6 text-${category.color}-400`} />
                </div>
                <h3 className="text-lg font-semibold text-white mb-4">{category.title}</h3>
                <ul className="space-y-3">
                  {category.items.map((item) => (
                    <li key={item.name} className="flex items-start gap-3">
                      <ChevronRight className="w-4 h-4 text-white/30 mt-0.5 flex-shrink-0" />
                      <div>
                        <span className="text-white/90 text-sm font-medium">{item.name}</span>
                        <p className="text-white/40 text-xs">{item.desc}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section className="py-24 bg-white/[0.01]">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">System Architecture</h2>
            <p className="text-white/60 max-w-2xl mx-auto">
              Full-stack application with decoupled frontend, REST API backend, and multi-stage ML pipeline
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="p-8 rounded-3xl bg-white/[0.02] border border-white/5"
          >
            <div className="grid md:grid-cols-3 gap-8">
              {/* Frontend */}
              <div className="text-center">
                <div className="w-16 h-16 mx-auto rounded-2xl bg-blue-500/10 flex items-center justify-center mb-4">
                  <Globe className="w-8 h-8 text-blue-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">Frontend</h3>
                <p className="text-white/50 text-sm mb-4">Next.js 14 + TypeScript</p>
                <ul className="text-left text-sm space-y-2 text-white/60">
                  <li>• Video upload with drag & drop</li>
                  <li>• Real-time job status polling</li>
                  <li>• Analysis options configuration</li>
                  <li>• Results visualization</li>
                </ul>
              </div>

              {/* Backend */}
              <div className="text-center">
                <div className="w-16 h-16 mx-auto rounded-2xl bg-green-500/10 flex items-center justify-center mb-4">
                  <Server className="w-8 h-8 text-green-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">Backend API</h3>
                <p className="text-white/50 text-sm mb-4">FastAPI + Python 3.11</p>
                <ul className="text-left text-sm space-y-2 text-white/60">
                  <li>• RESTful API endpoints</li>
                  <li>• Background task processing</li>
                  <li>• Job queue management</li>
                  <li>• Static file serving</li>
                </ul>
              </div>

              {/* ML Pipeline */}
              <div className="text-center">
                <div className="w-16 h-16 mx-auto rounded-2xl bg-purple-500/10 flex items-center justify-center mb-4">
                  <Brain className="w-8 h-8 text-purple-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">ML Pipeline</h3>
                <p className="text-white/50 text-sm mb-4">PyTorch + Roboflow</p>
                <ul className="text-left text-sm space-y-2 text-white/60">
                  <li>• Player detection (YOLOv8)</li>
                  <li>• Segmentation tracking (SAM2)</li>
                  <li>• Team classification (SigLIP)</li>
                  <li>• Homography transformation</li>
                </ul>
              </div>
            </div>

            {/* Flow arrows */}
            <div className="mt-8 pt-8 border-t border-white/5">
              <h4 className="text-center text-white/50 text-sm mb-4">Processing Pipeline</h4>
              <div className="flex items-center justify-center gap-4 flex-wrap text-sm">
                {[
                  'Video Upload',
                  'Frame Extraction',
                  'Player Detection',
                  'SAM2 Tracking',
                  'Team Classification',
                  'Tactical View',
                  'Output Video'
                ].map((step, i) => (
                  <div key={step} className="flex items-center gap-2">
                    <span className="px-3 py-1.5 rounded-lg bg-orange-500/10 text-orange-400 whitespace-nowrap">
                      {step}
                    </span>
                    {i < 6 && <ChevronRight className="w-4 h-4 text-white/20" />}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* API Endpoints */}
      <section className="py-24">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">API Endpoints</h2>
            <p className="text-white/60">RESTful API design for video analysis operations</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="overflow-hidden rounded-2xl border border-white/10"
          >
            <table className="w-full">
              <thead>
                <tr className="bg-white/5">
                  <th className="px-6 py-4 text-left text-sm font-medium text-white/70">Method</th>
                  <th className="px-6 py-4 text-left text-sm font-medium text-white/70">Endpoint</th>
                  <th className="px-6 py-4 text-left text-sm font-medium text-white/70">Description</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {[
                  { method: 'GET', endpoint: '/', desc: 'Health check and API info' },
                  { method: 'POST', endpoint: '/api/upload', desc: 'Upload video for analysis' },
                  { method: 'GET', endpoint: '/api/status/{job_id}', desc: 'Poll job processing status' },
                  { method: 'GET', endpoint: '/api/results/{job_id}', desc: 'Retrieve analysis results' },
                  { method: 'GET', endpoint: '/api/jobs', desc: 'List all jobs (debug)' },
                  { method: 'DELETE', endpoint: '/api/jobs/{job_id}', desc: 'Delete job and cleanup outputs' },
                ].map((api) => (
                  <tr key={api.endpoint} className="hover:bg-white/[0.02] transition-colors">
                    <td className="px-6 py-4">
                      <span className={`px-2 py-1 rounded text-xs font-mono ${
                        api.method === 'GET' ? 'bg-green-500/10 text-green-400' :
                        api.method === 'POST' ? 'bg-blue-500/10 text-blue-400' :
                        'bg-red-500/10 text-red-400'
                      }`}>
                        {api.method}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-white/80">{api.endpoint}</td>
                    <td className="px-6 py-4 text-sm text-white/60">{api.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </motion.div>
        </div>
      </section>

      {/* Challenges */}
      <section className="py-24 bg-white/[0.01]">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">Technical Challenges</h2>
            <p className="text-white/60 max-w-2xl mx-auto">
              Complex problems solved during development
            </p>
          </motion.div>

          <div className="space-y-6">
            {challenges.map((challenge, i) => (
              <motion.div
                key={challenge.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="p-8 rounded-2xl bg-white/[0.02] border border-white/5"
              >
                <h3 className="text-xl font-semibold text-white mb-6">{challenge.title}</h3>
                <div className="grid md:grid-cols-2 gap-8">
                  <div>
                    <div className="flex items-center gap-2 text-red-400 text-sm font-medium mb-3">
                      <div className="w-2 h-2 rounded-full bg-red-400" />
                      Problem
                    </div>
                    <p className="text-white/60">{challenge.problem}</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-2 text-green-400 text-sm font-medium mb-3">
                      <div className="w-2 h-2 rounded-full bg-green-400" />
                      Solution
                    </div>
                    <p className="text-white/60">{challenge.solution}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Code Metrics */}
      <section className="py-24">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">Project Metrics</h2>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              { icon: FileCode, label: 'Backend Python Files', value: '21 files', sub: '~3,150 LOC' },
              { icon: Code2, label: 'Frontend TypeScript', value: '2 files', sub: '~1,450 LOC' },
              { icon: Video, label: 'Demo Videos', value: '9 videos', sub: '~76 MB outputs' },
              { icon: ImageIcon, label: 'Sample Frames', value: '18 frames', sub: 'Key frame exports' },
              { icon: Brain, label: 'ML Modules', value: '6 modules', sub: '1,800+ LOC' },
              { icon: Database, label: 'Test Scripts', value: '12 scripts', sub: 'Development tests' },
            ].map((metric, i) => (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                className="p-6 rounded-2xl bg-white/[0.02] border border-white/5 text-center"
              >
                <metric.icon className="w-8 h-8 text-orange-500 mx-auto mb-4" />
                <div className="text-2xl font-bold text-white mb-1">{metric.value}</div>
                <div className="text-white/70 text-sm mb-1">{metric.label}</div>
                <div className="text-white/40 text-xs">{metric.sub}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* What I'd Do Differently */}
      <section className="py-24 bg-white/[0.01]">
        <div className="max-w-4xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-white mb-8 text-center">Lessons Learned</h2>

            <div className="p-8 rounded-2xl bg-gradient-to-br from-orange-500/10 to-transparent border border-orange-500/20">
              <ul className="space-y-4 text-white/70">
                <li className="flex items-start gap-3">
                  <ChevronRight className="w-5 h-5 text-orange-500 flex-shrink-0 mt-0.5" />
                  <span><strong className="text-white">Database from Day 1:</strong> Would integrate PostgreSQL earlier instead of in-memory job storage to enable job persistence and better scaling.</span>
                </li>
                <li className="flex items-start gap-3">
                  <ChevronRight className="w-5 h-5 text-orange-500 flex-shrink-0 mt-0.5" />
                  <span><strong className="text-white">Model Caching:</strong> Implement warm model pools to reduce first-request latency when loading large models like SAM2.</span>
                </li>
                <li className="flex items-start gap-3">
                  <ChevronRight className="w-5 h-5 text-orange-500 flex-shrink-0 mt-0.5" />
                  <span><strong className="text-white">GPU Batching:</strong> Process multiple frames in batches rather than sequentially for significant speed improvements on GPU.</span>
                </li>
                <li className="flex items-start gap-3">
                  <ChevronRight className="w-5 h-5 text-orange-500 flex-shrink-0 mt-0.5" />
                  <span><strong className="text-white">Video Streaming:</strong> Implement HLS/DASH streaming for large output videos instead of serving complete files.</span>
                </li>
              </ul>
            </div>
          </motion.div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-white mb-6">See It In Action</h2>
            <p className="text-white/60 mb-8 max-w-2xl mx-auto">
              Upload your own basketball footage and experience AI-powered game analysis firsthand.
            </p>
            <div className="flex items-center justify-center gap-4">
              <Link
                href="/"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-orange-500 to-orange-600 text-white font-medium text-lg hover:shadow-lg hover:shadow-orange-500/30 transition-all"
              >
                <Zap className="w-6 h-6" />
                Launch Demo
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-8">
        <div className="max-w-6xl mx-auto px-6 text-center text-white/40 text-sm">
          Built with Next.js, FastAPI, PyTorch, and Roboflow
        </div>
      </footer>
    </main>
  )
}

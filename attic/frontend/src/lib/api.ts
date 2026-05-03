import axios, { AxiosError, AxiosInstance } from 'axios';

// API base URL - can be overridden via environment variable
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

// Types
export interface User {
  id: string;
  email: string;
  full_name: string | null;
  role: string;
  plan: string;
  usage_minutes_remaining: number;
  is_active: boolean;
  created_at: string;
}

export interface Team {
  id: string;
  name: string;
  color: string;
  roster: Record<string, string>;
  logo_url?: string;
  league?: string;
  created_at: string;
}

export interface TeamConfig {
  name: string;
  color: string;
  roster: Record<string, string>;
}

export interface ProcessingConfig {
  max_duration_seconds?: number;
  start_time_seconds?: number;
  enable_segmentation?: boolean;
  enable_jersey_detection?: boolean;
  enable_tactical_view?: boolean;
}

export interface VisualizationConfig {
  mask_display?: 'all' | 'team_a' | 'team_b' | 'selected' | 'none';
  selected_player_ids?: number[];
  show_bounding_boxes?: boolean;
  show_player_names?: boolean;
  show_jersey_numbers?: boolean;
  tactical_view_position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left' | 'none';
  tactical_view_scale?: number;
}

export interface DetectedPlayer {
  id: number;
  team: string;
  jersey_number?: string;
  player_name?: string;
  frames_tracked: number;
}

export interface JobVideoUrls {
  stage1_detection?: string;
  stage2_tracking?: string;
  stage3_segmentation?: string;
  stage4_teams?: string;
  stage5_jerseys?: string;
  final?: string;
}

export interface JobSummary {
  id: string;
  status: string;
  progress: number;
  team_a_name?: string;
  team_b_name?: string;
  input_video_filename?: string;
  created_at: string;
}

export interface JobDetail {
  id: string;
  status: string;
  progress: number;
  current_stage?: string;
  error_message?: string;
  team_a: TeamConfig;
  team_b: TeamConfig;
  processing_config: ProcessingConfig;
  visualization_config: VisualizationConfig;
  detected_players?: DetectedPlayer[];
  video_urls?: JobVideoUrls;
  input_video_filename?: string;
  input_video_duration_seconds?: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  processing_time_seconds?: number;
}

export interface JobListResponse {
  jobs: JobSummary[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

// API Error type
export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string
  ) {
    super(detail);
    this.name = 'ApiError';
  }
}

// Create axios instance
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_BASE_URL,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor - add auth token
  client.interceptors.request.use((config) => {
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('access_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  });

  // Response interceptor - handle errors
  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError<{ detail: string }>) => {
      if (error.response) {
        const detail = error.response.data?.detail || 'An error occurred';
        throw new ApiError(error.response.status, detail);
      }
      throw new ApiError(500, 'Network error');
    }
  );

  return client;
};

const apiClient = createApiClient();

// ============================================================================
// Auth API
// ============================================================================

export const authApi = {
  async register(email: string, password: string, fullName?: string): Promise<AuthResponse> {
    const { data } = await apiClient.post<AuthResponse>('/auth/register', {
      email,
      password,
      full_name: fullName,
    });
    if (typeof window !== 'undefined') {
      localStorage.setItem('access_token', data.access_token);
    }
    return data;
  },

  async login(email: string, password: string): Promise<AuthResponse> {
    const { data } = await apiClient.post<AuthResponse>('/auth/login', {
      email,
      password,
    });
    if (typeof window !== 'undefined') {
      localStorage.setItem('access_token', data.access_token);
    }
    return data;
  },

  async getMe(): Promise<User> {
    const { data } = await apiClient.get<User>('/auth/me');
    return data;
  },

  logout() {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('access_token');
    }
  },
};

// ============================================================================
// Teams API
// ============================================================================

export const teamsApi = {
  async list(): Promise<Team[]> {
    const { data } = await apiClient.get<Team[]>('/teams/');
    return data;
  },

  async create(team: Omit<Team, 'id' | 'created_at'>): Promise<Team> {
    const { data } = await apiClient.post<Team>('/teams/', team);
    return data;
  },

  async get(teamId: string): Promise<Team> {
    const { data } = await apiClient.get<Team>(`/teams/${teamId}`);
    return data;
  },

  async update(teamId: string, team: Partial<Team>): Promise<Team> {
    const { data } = await apiClient.patch<Team>(`/teams/${teamId}`, team);
    return data;
  },

  async delete(teamId: string): Promise<void> {
    await apiClient.delete(`/teams/${teamId}`);
  },
};

// ============================================================================
// Jobs API
// ============================================================================

export const jobsApi = {
  async list(page = 1, perPage = 20, statusFilter?: string): Promise<JobListResponse> {
    const params = new URLSearchParams({
      page: page.toString(),
      per_page: perPage.toString(),
    });
    if (statusFilter) {
      params.append('status_filter', statusFilter);
    }
    const { data } = await apiClient.get<JobListResponse>(`/jobs/?${params}`);
    return data;
  },

  async create(
    video: File,
    teamA: TeamConfig,
    teamB: TeamConfig,
    processing?: ProcessingConfig,
    visualization?: VisualizationConfig
  ): Promise<{ id: string; status: string }> {
    const formData = new FormData();
    formData.append('video', video);
    formData.append('team_a', JSON.stringify(teamA));
    formData.append('team_b', JSON.stringify(teamB));
    if (processing) {
      formData.append('processing', JSON.stringify(processing));
    }
    if (visualization) {
      formData.append('visualization', JSON.stringify(visualization));
    }

    const { data } = await apiClient.post('/jobs/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return data;
  },

  async get(jobId: string): Promise<JobDetail> {
    const { data } = await apiClient.get<JobDetail>(`/jobs/${jobId}`);
    return data;
  },

  async getStatus(jobId: string): Promise<{ id: string; status: string; progress: number; current_stage?: string }> {
    const { data } = await apiClient.get(`/jobs/${jobId}/status`);
    return data;
  },

  async rerender(jobId: string, visualization: VisualizationConfig): Promise<{ id: string; status: string }> {
    const { data } = await apiClient.post(`/jobs/${jobId}/rerender`, { visualization });
    return data;
  },

  async delete(jobId: string): Promise<void> {
    await apiClient.delete(`/jobs/${jobId}`);
  },

  async cancel(jobId: string): Promise<{ id: string; status: string }> {
    const { data } = await apiClient.post(`/jobs/${jobId}/cancel`);
    return data;
  },
};

// Export the api client for custom requests
export { apiClient };

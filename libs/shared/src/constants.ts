// Shared constants across the RAG Proactive Lab ecosystem

export const AGENT_NAMES = {
  AVI: 'avi',
  HASE: 'hase',
  PIA: 'pia',
  TIR: 'tir',
  POSTVENTA: 'postventa',
  GUARDIAN: 'guardian',
} as const;

export const SEVERITY_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical',
} as const;

export const CONTRACT_STATUS = {
  ACTIVE: 'active',
  INACTIVE: 'inactive',
  EXPIRED: 'expired',
  PENDING: 'pending',
} as const;

export const API_ENDPOINTS = {
  HEALTH: '/health',
  AGENTS: '/agents',
  DRIVERS: '/drivers',
  ALERTS: '/alerts',
  REPORTS: '/reports',
} as const;

export const DEFAULT_PAGINATION = {
  PAGE: 1,
  LIMIT: 20,
  MAX_LIMIT: 100,
} as const;
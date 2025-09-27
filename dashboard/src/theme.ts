const palette = {
  bgPrimary: '#0d1117',
  bgSecondary: '#161b22',
  bgTertiary: '#1f2937',
  textPrimary: '#f0f6fc',
  textSecondary: '#b1bac4',
  textMuted: '#8b949e',
  borderPrimary: '#21262d',
  borderSoft: 'rgba(33, 38, 45, 0.6)',
  primary: '#00bfbf',
  primarySoft: 'rgba(0, 191, 191, 0.16)',
  success: '#22c55e',
  successSoft: 'rgba(34, 197, 94, 0.16)',
  warning: '#f59e0b',
  warningSoft: 'rgba(245, 158, 11, 0.18)',
  danger: '#ef4444',
  dangerSoft: 'rgba(239, 68, 68, 0.18)',
};

export const theme = {
  colors: {
    bg: palette.bgPrimary,
    surface: palette.bgSecondary,
    card: palette.bgTertiary,
    textPrimary: palette.textPrimary,
    textSecondary: palette.textSecondary,
    textMuted: palette.textMuted,
    border: palette.borderPrimary,
    borderSoft: palette.borderSoft,
    primary: palette.primary,
    primarySoft: palette.primarySoft,
    success: palette.success,
    successSoft: palette.successSoft,
    warning: palette.warning,
    warningSoft: palette.warningSoft,
    danger: palette.danger,
    dangerSoft: palette.dangerSoft,
  },
  radii: {
    sm: '6px',
    md: '12px',
    lg: '18px',
  },
  shadow: '0 42px 88px -36px rgba(8, 13, 28, 0.78)',
  fonts: {
    system:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif",
    display: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    mono: "'SF Mono', 'JetBrains Mono', 'Source Code Pro', monospace",
  },
};

export type AppTheme = typeof theme;

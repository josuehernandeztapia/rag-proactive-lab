import styled from 'styled-components';

type Tone = 'neutral' | 'success' | 'warning' | 'danger';

export const Badge = styled.span<{ tone?: Tone }>`
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  background: ${({ tone = 'neutral', theme }) => {
    const map: Record<Tone, string> = {
      neutral: theme.colors.primarySoft,
      success: theme.colors.successSoft,
      warning: theme.colors.warningSoft,
      danger: theme.colors.dangerSoft,
    };
    return map[tone];
  }};
  color: ${({ tone = 'neutral', theme }) => {
    const map: Record<Tone, string> = {
      neutral: theme.colors.primary,
      success: theme.colors.success,
      warning: theme.colors.warning,
      danger: theme.colors.danger,
    };
    return map[tone];
  }};
`;

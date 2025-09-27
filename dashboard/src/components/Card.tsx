import styled from 'styled-components';
import type { ReactNode } from 'react';

const CardWrapper = styled.section`
  background: radial-gradient(circle at top, rgba(0, 191, 191, 0.05), rgba(13, 17, 23, 0) 65%),
    ${({ theme }) => theme.colors.card};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  box-shadow: ${({ theme }) => theme.shadow};
  padding: 24px;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 44px 96px -44px rgba(4, 10, 24, 0.85);
    border-color: ${({ theme }) => theme.colors.borderSoft};
  }
`;

const CardHeader = styled.header`
  margin-bottom: 16px;

  h2 {
    margin: 0;
    font-size: 18px;
  }

  p {
    margin: 4px 0 0;
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: 14px;
  }
`;

type CardProps = {
  children: ReactNode;
  title?: string;
  subtitle?: string;
};

export function Card({ children, title, subtitle }: CardProps) {
  return (
    <CardWrapper>
      {title ? (
        <CardHeader>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </CardHeader>
      ) : null}
      {children}
    </CardWrapper>
  );
}

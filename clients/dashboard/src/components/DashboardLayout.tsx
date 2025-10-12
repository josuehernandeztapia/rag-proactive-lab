import styled from 'styled-components';
import type { ReactNode } from 'react';

const Wrapper = styled.div`
  min-height: 100vh;
  display: flex;
  background: ${({ theme }) => theme.colors.bg};
`;

const Sidebar = styled.aside`
  width: 260px;
  padding: 32px 24px;
  background: linear-gradient(180deg, rgba(15, 118, 110, 0.12), transparent 60%),
    ${({ theme }) => theme.colors.surface};
  border-right: 1px solid ${({ theme }) => theme.colors.border};
  display: flex;
  flex-direction: column;
  gap: 24px;
`;

const Brand = styled.div`
  font-weight: 700;
  font-size: 18px;
`;

const Nav = styled.nav`
  display: grid;
  gap: 12px;

  a {
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;

    &:hover {
      color: ${({ theme }) => theme.colors.textPrimary};
    }
  }
`;

const Content = styled.main`
  flex: 1;
  padding: 32px 40px 48px;
  background: linear-gradient(180deg, rgba(15, 118, 110, 0.08), transparent 40%),
    ${({ theme }) => theme.colors.bg};
  display: flex;
  flex-direction: column;
  gap: 24px;
`;

type DashboardLayoutProps = {
  sidebarLinks: { href: string; label: string }[];
  children: ReactNode;
};

export function DashboardLayout({ sidebarLinks, children }: DashboardLayoutProps) {
  return (
    <Wrapper>
      <Sidebar>
        <Brand>Demo Protección</Brand>
        <Nav>
          {sidebarLinks.map((link) => (
            <a key={link.href} href={link.href}>
              {link.label}
            </a>
          ))}
        </Nav>
      </Sidebar>
      <Content>{children}</Content>
    </Wrapper>
  );
}

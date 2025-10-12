import { createGlobalStyle } from 'styled-components';

export const GlobalStyles = createGlobalStyle`
  :root {
    color-scheme: dark;
    font-family: ${({ theme }) => theme.fonts.display};
    font-size: 16px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  *, *::before, *::after {
    box-sizing: border-box;
  }

  body {
    margin: 0;
    background: ${({ theme }) => theme.colors.bg};
    color: ${({ theme }) => theme.colors.textPrimary};
    font-family: ${({ theme }) => theme.fonts.display};
  }

  a {
    color: inherit;
    text-decoration: none;
  }

  button {
    font-family: inherit;
  }

  code, pre {
    font-family: ${({ theme }) => theme.fonts.mono};
  }
`;

import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the AeroSense landing page', () => {
  render(<App />);
  expect(screen.getByRole('link', { name: /AeroSense home/i })).toBeInTheDocument();
});

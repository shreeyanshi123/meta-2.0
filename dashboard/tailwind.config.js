/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        tribunal: {
          bg: '#0b1020',
          surface: '#0f172a',
          'surface-light': '#1e293b',
          border: 'rgba(255, 255, 255, 0.10)',
          gold: '#f1c27d',
          'gold-dim': '#c9a05c',
          cyan: '#22d3ee',
          rose: '#fb7185',
          amber: '#fbbf24',
          emerald: '#34d399',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'tribunal-gradient': 'linear-gradient(135deg, #0b1020 0%, #0f172a 50%, #131b2e 100%)',
        'gold-gradient': 'linear-gradient(135deg, #f1c27d, #d4a254)',
        'glass-shine': 'linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 100%)',
      },
      boxShadow: {
        'glass': '0 8px 32px rgba(0, 0, 0, 0.3)',
        'glass-lg': '0 16px 64px rgba(0, 0, 0, 0.4)',
        'glow-gold': '0 0 20px rgba(241, 194, 125, 0.3)',
        'glow-cyan': '0 0 20px rgba(34, 211, 238, 0.3)',
        'glow-rose': '0 0 20px rgba(251, 113, 133, 0.3)',
      },
      animation: {
        'gavel-strike': 'gavelStrike 0.6s cubic-bezier(0.22, 1, 0.36, 1)',
        'pulse-rose': 'pulseRose 1s ease-in-out',
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'flip': 'flip 0.6s ease-in-out',
      },
      keyframes: {
        gavelStrike: {
          '0%': { transform: 'rotate(-30deg) scale(1.2)', opacity: '0.5' },
          '50%': { transform: 'rotate(5deg) scale(0.95)' },
          '100%': { transform: 'rotate(0deg) scale(1)', opacity: '1' },
        },
        pulseRose: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(251, 113, 133, 0)' },
          '50%': { boxShadow: '0 0 24px 4px rgba(251, 113, 133, 0.4)' },
        },
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        flip: {
          '0%': { transform: 'rotateY(0deg)' },
          '50%': { transform: 'rotateY(90deg)' },
          '100%': { transform: 'rotateY(0deg)' },
        },
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
}

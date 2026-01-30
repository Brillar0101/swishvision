/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Primary brand colors
        blacktop: '#0D0D0D',
        sideline: '#F8F7F4',

        // Court orange - primary accent
        court: {
          DEFAULT: '#E85D04',
          50: '#FEF6F0',
          100: '#FEF3E7',
          200: '#FDDCC4',
          300: '#FBC49A',
          400: '#F79355',
          500: '#E85D04',
          600: '#D4520A',
          700: '#B04308',
          800: '#8C3506',
          900: '#6D2A05',
        },

        // Baseline gray
        baseline: {
          DEFAULT: '#6B7280',
          50: '#F9FAFB',
          100: '#F3F4F6',
          200: '#E5E7EB',
          300: '#D1D5DB',
          400: '#9CA3AF',
          500: '#6B7280',
          600: '#4B5563',
          700: '#374151',
          800: '#1F2937',
          900: '#111827',
        },

        // Secondary accent (navy)
        away: {
          DEFAULT: '#1E3A5F',
          light: '#2D5A8A',
          dark: '#0F1D30',
        },

        // Court wood
        hardwood: '#D4A574',
      },

      fontFamily: {
        display: ['Instrument Sans', 'system-ui', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
      },

      fontSize: {
        'display-1': ['48px', { lineHeight: '1.1', letterSpacing: '-0.02em', fontWeight: '700' }],
        'display-2': ['32px', { lineHeight: '1.2', letterSpacing: '-0.01em', fontWeight: '600' }],
        'display-3': ['24px', { lineHeight: '1.3', letterSpacing: '0', fontWeight: '600' }],
      },

      boxShadow: {
        'card': '0 1px 3px rgba(13,13,13,0.04), 0 4px 12px rgba(13,13,13,0.03)',
        'card-hover': '0 4px 6px rgba(13,13,13,0.05), 0 12px 24px rgba(13,13,13,0.06)',
        'elevated': '0 4px 6px rgba(13,13,13,0.05), 0 12px 24px rgba(13,13,13,0.08)',
        'button': '0 2px 8px rgba(232,93,4,0.25)',
        'button-hover': '0 4px 12px rgba(232,93,4,0.35)',
        'input-focus': '0 0 0 3px rgba(232,93,4,0.1)',
      },

      borderRadius: {
        'none': '0',
        'sm': '4px',
        'DEFAULT': '8px',
        'md': '12px',
        'lg': '16px',
        'xl': '24px',
        'card': '12px',
        'modal': '16px',
        'full': '9999px',
      },

      animation: {
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'scale-in': 'scaleIn 0.2s ease-out',
        'bounce-subtle': 'bounceSubtle 0.6s ease-in-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 2s linear infinite',
      },

      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        bounceSubtle: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-5px)' },
        },
      },

      transitionTimingFunction: {
        'bounce-in': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
      },
    },
  },
  plugins: [],
}

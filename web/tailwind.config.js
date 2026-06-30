/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Manrope", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        agent: {
          50: "#ecfdf5",
          400: "#34d399",
          500: "#10b981",
          600: "#059669",
        },
      },
    },
  },
  plugins: [],
};

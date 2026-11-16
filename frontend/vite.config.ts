import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('lucide-react')) {
              return 'vendor-icons';
            }
            if (id.includes('react') || id.includes('react-dom')) {
              return 'vendor-react';
            }
            return 'vendor-misc';
          }
          if (id.endsWith('.json')) {
            if (id.includes('coordinate_ai_model')) {
              return 'model-coordinate-ai';
            }
            if (id.includes('isl_alphabet_model') || id.includes('single_hand_alphabet_model')) {
              return 'model-alphabet-legacy';
            }
            if (id.includes('letter_a_specialist_model') || id.includes('dynamic_alphabet_model') || id.includes('titan_spatial_model')) {
              return 'model-specialists';
            }
          }
        },
      },
    },
    chunkSizeWarningLimit: 1600,
  },
})

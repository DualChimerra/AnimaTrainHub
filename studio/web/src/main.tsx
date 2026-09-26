import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { DialogProvider } from './components/Dialog'
import { ErrorBoundary } from './components/ErrorBoundary'
import RuntimeModeGate from './components/RuntimeModeGate'
import { ToastProvider } from './components/Toast'
import { installGlobalErrorHandlers } from './lib/errors/setup'
import { RuntimeModeProvider } from './lib/RuntimeMode'
import { SettingsDataProvider } from './lib/SettingsData'
import { SettingsDrawerProvider } from './lib/SettingsDrawer'
import { initTheme } from './lib/theme'
import { installSoundFeedback } from './lib/sound'
import SelectHost from './components/ds/SelectHost'
import './i18n'
import './index.css'

// ADR-0009 PR-3 C2: window.onerror + unhandledrejection, both routes captured -> /api/client-errors
installGlobalErrorHandlers()

initTheme()
installSoundFeedback()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ToastProvider>
        <DialogProvider>
          <SettingsDataProvider>
            <SettingsDrawerProvider>
              {/* RuntimeModeProvider wraps outside App: the Topbar badge / Settings
                  section both need to read the mode, and the Gate needs to sit as
                  an overlay above the whole app (must be answered once on first
                  visit). */}
              <RuntimeModeProvider>
                <App />
                <RuntimeModeGate />
                <SelectHost />
              </RuntimeModeProvider>
            </SettingsDrawerProvider>
          </SettingsDataProvider>
        </DialogProvider>
      </ToastProvider>
    </ErrorBoundary>
  </React.StrictMode>,
)

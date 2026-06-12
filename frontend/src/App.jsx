import { useState } from 'react'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'

export default function App() {
  const [started, setStarted] = useState(false)

  if (!started) return <Landing onStart={() => setStarted(true)} />
  return <Dashboard />
}

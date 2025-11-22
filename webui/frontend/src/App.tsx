import { useState } from 'react'
import { Send, Loader2, AlertCircle, CheckCircle2, Database, Globe, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Alert, AlertDescription } from '@/components/ui/alert'
import './App.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources_used?: string[]
  citations?: string[]
  latency_ms?: number
  fast_path?: boolean
}

interface SystemStatus {
  qdrant: boolean
  elasticsearch: boolean
  redis: boolean
  embeddings: boolean
  overall: boolean
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [strictLocal, setStrictLocal] = useState(false)
  const [webSearchEnabled, setWebSearchEnabled] = useState(true)
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  // Check system status on mount
  useState(() => {
    fetch(`${API_URL}/api/status`)
      .then(res => res.json())
      .then(setSystemStatus)
      .catch(() => setSystemStatus(null))
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage: Message = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          strict_local: strictLocal,
          fast: false
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to get response')
      }

      const data = await response.json()
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
        sources_used: data.sources_used,
        citations: data.citations,
        latency_ms: data.latency_ms,
        fast_path: data.fast_path
      }
      
      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const getSourceIcon = (source: string) => {
    if (source === 'local_knowledge_base') return <Database className="w-4 h-4" />
    if (source === 'web_search') return <Globe className="w-4 h-4" />
    return <Zap className="w-4 h-4" />
  }

  const getSourceColor = (source: string) => {
    if (source === 'local_knowledge_base') return 'bg-blue-100 text-blue-800 border-blue-200'
    if (source === 'web_search') return 'bg-green-100 text-green-800 border-green-200'
    return 'bg-orange-100 text-orange-800 border-orange-200'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">RAG System</h1>
            <p className="text-sm text-slate-600">Intelligent Question Answering</p>
          </div>
          
          {/* System Status */}
          {systemStatus && (
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${systemStatus.overall ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm text-slate-600">
                {systemStatus.overall ? 'All Systems Operational' : 'System Issues Detected'}
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Settings Sidebar */}
          <aside className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label htmlFor="strict-local" className="text-sm">
                    Strict Local Mode
                  </Label>
                  <Switch
                    id="strict-local"
                    checked={strictLocal}
                    onCheckedChange={setStrictLocal}
                  />
                </div>
                <p className="text-xs text-slate-500">
                  Only use local knowledge base, no external sources
                </p>

                <div className="border-t pt-4">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="web-search" className="text-sm">
                      Web Search
                    </Label>
                    <Switch
                      id="web-search"
                      checked={webSearchEnabled}
                      onCheckedChange={setWebSearchEnabled}
                      disabled={strictLocal}
                    />
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Enable web search for current information
                  </p>
                </div>

                {systemStatus && (
                  <div className="border-t pt-4 space-y-2">
                    <p className="text-sm font-medium">System Status</p>
                    <div className="space-y-1 text-xs">
                      <div className="flex items-center justify-between">
                        <span>Qdrant</span>
                        {systemStatus.qdrant ? (
                          <CheckCircle2 className="w-4 h-4 text-green-600" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-600" />
                        )}
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Elasticsearch</span>
                        {systemStatus.elasticsearch ? (
                          <CheckCircle2 className="w-4 h-4 text-green-600" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-600" />
                        )}
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Redis</span>
                        {systemStatus.redis ? (
                          <CheckCircle2 className="w-4 h-4 text-green-600" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-600" />
                        )}
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Embeddings</span>
                        {systemStatus.embeddings ? (
                          <CheckCircle2 className="w-4 h-4 text-green-600" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-600" />
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </aside>

          {/* Chat Area */}
          <div className="lg:col-span-3 space-y-4">
            {/* Messages */}
            <Card className="min-h-[500px] max-h-[600px] overflow-y-auto">
              <CardContent className="p-6 space-y-4">
                {messages.length === 0 && (
                  <div className="text-center py-12 text-slate-500">
                    <p className="text-lg font-medium">Welcome to RAG System</p>
                    <p className="text-sm mt-2">Ask me anything!</p>
                  </div>
                )}

                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-4 ${
                        message.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-white border border-slate-200'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{message.content}</p>

                      {message.role === 'assistant' && (
                        <div className="mt-3 space-y-2">
                          {/* Sources Used */}
                          {message.sources_used && message.sources_used.length > 0 && (
                            <div className="flex flex-wrap gap-2">
                              {message.sources_used.map((source, i) => (
                                <Badge
                                  key={i}
                                  variant="outline"
                                  className={`text-xs ${getSourceColor(source)}`}
                                >
                                  <span className="mr-1">{getSourceIcon(source)}</span>
                                  {source.replace(/_/g, ' ')}
                                </Badge>
                              ))}
                            </div>
                          )}

                          {/* Citations */}
                          {message.citations && message.citations.length > 0 && (
                            <div className="text-xs text-slate-600 space-y-1 border-t pt-2">
                              <p className="font-medium">Citations:</p>
                              {message.citations.map((citation, i) => (
                                <p key={i} className="pl-2">{citation}</p>
                              ))}
                            </div>
                          )}

                          {/* Metadata */}
                          <div className="flex items-center gap-3 text-xs text-slate-500 border-t pt-2">
                            {message.latency_ms && (
                              <span>{(message.latency_ms / 1000).toFixed(2)}s</span>
                            )}
                            {message.fast_path && (
                              <Badge variant="outline" className="text-xs">
                                Fast Path
                              </Badge>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="flex justify-start">
                    <div className="bg-white border border-slate-200 rounded-lg p-4">
                      <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Error Display */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Input Form */}
            <Card>
              <CardContent className="p-4">
                <form onSubmit={handleSubmit} className="flex gap-2">
                  <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask me anything..."
                    className="min-h-[60px] resize-none"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        handleSubmit(e)
                      }
                    }}
                  />
                  <Button
                    type="submit"
                    disabled={loading || !input.trim()}
                    className="self-end"
                  >
                    {loading ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Send className="w-4 h-4" />
                    )}
                  </Button>
                </form>
                <p className="text-xs text-slate-500 mt-2">
                  Press Enter to send, Shift+Enter for new line
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App

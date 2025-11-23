import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { Send, Loader2, AlertCircle, CheckCircle2, Database, Globe, Zap, RotateCcw, Upload, Link as LinkIcon, ChevronDown, ChevronUp, MessageSquare, FolderOpen, Activity } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Input } from '@/components/ui/input'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import './App.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources_used?: string[]
  citations?: string[]
  latency_ms?: number
  fast_path?: boolean
  tool_results?: any
}

interface SystemStatus {
  qdrant: boolean
  elasticsearch: boolean
  redis: boolean
  embeddings: boolean
  overall: boolean
}

interface KBStats {
  document_count: number
  chunk_count: number
  last_ingested_at: string | null
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [strictLocal, setStrictLocal] = useState(false)
  const [webSearchEnabled, setWebSearchEnabled] = useState(true)
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [kbStats, setKbStats] = useState<KBStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [expandedDetails, setExpandedDetails] = useState<number | null>(null)
  const [activeTab, setActiveTab] = useState<'chat' | 'kb'>('chat')
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null)
  const [urlInput, setUrlInput] = useState('')
  const [ingesting, setIngesting] = useState(false)
  const [ingestMessage, setIngestMessage] = useState<string | null>(null)
  const [statusDialogOpen, setStatusDialogOpen] = useState(false)

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  useEffect(() => {
    fetchSystemStatus()
    fetchKBStats()
    const interval = setInterval(() => {
      fetchSystemStatus()
      if (activeTab === 'kb') {
        fetchKBStats()
      }
    }, 10000)
    return () => clearInterval(interval)
  }, [activeTab])

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/api/status`)
      const data = await res.json()
      setSystemStatus(data)
    } catch {
      setSystemStatus(null)
    }
  }

  const fetchKBStats = async () => {
    try {
      const res = await fetch(`${API_URL}/api/kb/stats`)
      const data = await res.json()
      setKbStats(data)
    } catch {
      setKbStats(null)
    }
  }

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
          fast: false,
          web_search: webSearchEnabled
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
        fast_path: data.fast_path,
        tool_results: data.tool_results
      }
      
      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleIngest = async () => {
    if (!uploadFiles && !urlInput.trim()) {
      setIngestMessage('Please select files or enter a URL')
      return
    }

    setIngesting(true)
    setIngestMessage(null)

    try {
      const formData = new FormData()
      
      if (uploadFiles) {
        Array.from(uploadFiles).forEach(file => {
          formData.append('files', file)
        })
      }
      
      if (urlInput.trim()) {
        formData.append('urls', urlInput)
      }

      const response = await fetch(`${API_URL}/api/ingest`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to ingest documents')
      }

      const data = await response.json()
      setIngestMessage(`Success! Ingested ${data.files_processed} file(s), created ${data.chunks_created || 0} chunks`)
      setUploadFiles(null)
      setUrlInput('')
      fetchKBStats()
      
      const fileInput = document.getElementById('file-upload') as HTMLInputElement
      if (fileInput) fileInput.value = ''
    } catch (err) {
      setIngestMessage(`Error: ${err instanceof Error ? err.message : 'An error occurred'}`)
    } finally {
      setIngesting(false)
    }
  }

  const getSourceIcon = (source: string) => {
    if (source === 'local_knowledge_base') return <Database className="w-3 h-3" />
    if (source === 'web_search') return <Globe className="w-3 h-3" />
    if (source === 'weather') return <Zap className="w-3 h-3" />
    if (source === 'finance') return <Zap className="w-3 h-3" />
    if (source === 'transport') return <Zap className="w-3 h-3" />
    if (source === 'attachments') return <Upload className="w-3 h-3" />
    return <Zap className="w-3 h-3" />
  }

  const getSourceColor = (source: string) => {
    if (source === 'local_knowledge_base') return 'bg-blue-100 text-blue-800 border-blue-200'
    if (source === 'web_search') return 'bg-green-100 text-green-800 border-green-200'
    return 'bg-orange-100 text-orange-800 border-orange-200'
  }

  const handleNewChat = () => {
    setMessages([])
    setError(null)
    setExpandedDetails(null)
    fetchSystemStatus()
  }

  const toggleDetails = (index: number) => {
    setExpandedDetails(expandedDetails === index ? null : index)
  }

  const renderToolDetails = (toolResults: any, sources: string[]) => {
    if (!toolResults || Object.keys(toolResults).length === 0) return null

    return (
      <div className="mt-3 space-y-2 border-t pt-2">
        {toolResults.weather && sources.includes('weather') && (
          <div className="bg-orange-50 border border-orange-200 rounded p-3">
            <p className="text-xs font-semibold text-orange-900 mb-2">Weather Information</p>
            <div className="text-xs text-orange-800 space-y-1">
              {toolResults.weather.location && <p><strong>Location:</strong> {toolResults.weather.location}</p>}
              {toolResults.weather.temperature && <p><strong>Temperature:</strong> {toolResults.weather.temperature}</p>}
              {toolResults.weather.condition && <p><strong>Condition:</strong> {toolResults.weather.condition}</p>}
              {toolResults.weather.humidity && <p><strong>Humidity:</strong> {toolResults.weather.humidity}</p>}
            </div>
          </div>
        )}

        {toolResults.finance && sources.includes('finance') && (
          <div className="bg-orange-50 border border-orange-200 rounded p-3">
            <p className="text-xs font-semibold text-orange-900 mb-2">Financial Data</p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-orange-200">
                    <th className="text-left py-1 px-2">Ticker</th>
                    <th className="text-right py-1 px-2">Price</th>
                    <th className="text-right py-1 px-2">Change</th>
                    <th className="text-right py-1 px-2">Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.isArray(toolResults.finance) ? toolResults.finance.map((item: any, i: number) => (
                    <tr key={i} className="border-b border-orange-100">
                      <td className="py-1 px-2 font-medium">{item.ticker || item.symbol}</td>
                      <td className="text-right py-1 px-2">${item.price || item.current_price}</td>
                      <td className={`text-right py-1 px-2 ${(item.change || 0) >= 0 ? 'text-green-700' : 'text-red-700'}`}>
                        {item.change || item.daily_change || 'N/A'}
                      </td>
                      <td className="text-right py-1 px-2 text-slate-600">{item.timestamp || 'N/A'}</td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan={4} className="py-1 px-2 text-center text-slate-600">
                        {JSON.stringify(toolResults.finance)}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {toolResults.transport && sources.includes('transport') && (
          <div className="bg-orange-50 border border-orange-200 rounded p-3">
            <p className="text-xs font-semibold text-orange-900 mb-2">Route Information</p>
            <div className="text-xs text-orange-800 space-y-1">
              {toolResults.transport.origin && <p><strong>From:</strong> {toolResults.transport.origin}</p>}
              {toolResults.transport.destination && <p><strong>To:</strong> {toolResults.transport.destination}</p>}
              {toolResults.transport.distance && <p><strong>Distance:</strong> {toolResults.transport.distance}</p>}
              {toolResults.transport.duration && <p><strong>Duration:</strong> {toolResults.transport.duration}</p>}
            </div>
          </div>
        )}

        {toolResults.web_search && sources.includes('web_search') && Array.isArray(toolResults.web_search) && (
          <div className="bg-green-50 border border-green-200 rounded p-3">
            <p className="text-xs font-semibold text-green-900 mb-2">Web Search Results</p>
            <div className="space-y-2">
              {toolResults.web_search.slice(0, 5).map((result: any, i: number) => (
                <div key={i} className="text-xs">
                  <a 
                    href={result.url || result.link} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-green-700 hover:text-green-900 font-medium underline"
                  >
                    {result.title}
                  </a>
                  {result.snippet && <p className="text-slate-600 mt-1">{result.snippet}</p>}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">RAG System</h1>
            <p className="text-sm text-slate-600">Intelligent Question Answering with Knowledge Base</p>
          </div>
          
          <div className="flex items-center gap-4">
            {messages.length > 0 && activeTab === 'chat' && (
              <Button variant="outline" size="sm" onClick={handleNewChat} className="flex items-center gap-2">
                <RotateCcw className="w-4 h-4" />
                New Chat
              </Button>
            )}
            
            <Dialog open={statusDialogOpen} onOpenChange={setStatusDialogOpen}>
              <DialogTrigger asChild>
                <button className="flex items-center gap-2 hover:bg-slate-100 px-3 py-2 rounded-lg transition-colors">
                  <div className={`w-2 h-2 rounded-full ${systemStatus?.overall ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className="text-sm text-slate-600">
                    {systemStatus?.overall ? 'All Systems Operational' : 'System Issues Detected'}
                  </span>
                  <Activity className="w-4 h-4 text-slate-400" />
                </button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>System Status</DialogTitle>
                  <DialogDescription>Current status of all RAG system services</DialogDescription>
                </DialogHeader>
                <div className="space-y-3 py-4">
                  {systemStatus && (
                    <>
                      <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <Database className="w-5 h-5 text-slate-600" />
                          <div>
                            <p className="font-medium">Qdrant</p>
                            <p className="text-xs text-slate-500">Vector Database</p>
                          </div>
                        </div>
                        {systemStatus.qdrant ? <CheckCircle2 className="w-5 h-5 text-green-600" /> : <AlertCircle className="w-5 h-5 text-red-600" />}
                      </div>
                      <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <Database className="w-5 h-5 text-slate-600" />
                          <div>
                            <p className="font-medium">Elasticsearch</p>
                            <p className="text-xs text-slate-500">Full-text Search</p>
                          </div>
                        </div>
                        {systemStatus.elasticsearch ? <CheckCircle2 className="w-5 h-5 text-green-600" /> : <AlertCircle className="w-5 h-5 text-red-600" />}
                      </div>
                      <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <Database className="w-5 h-5 text-slate-600" />
                          <div>
                            <p className="font-medium">Redis</p>
                            <p className="text-xs text-slate-500">Cache & Session Store</p>
                          </div>
                        </div>
                        {systemStatus.redis ? <CheckCircle2 className="w-5 h-5 text-green-600" /> : <AlertCircle className="w-5 h-5 text-red-600" />}
                      </div>
                      <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <Zap className="w-5 h-5 text-slate-600" />
                          <div>
                            <p className="font-medium">Embeddings</p>
                            <p className="text-xs text-slate-500">BAAI/bge-m3 Model</p>
                          </div>
                        </div>
                        {systemStatus.embeddings ? <CheckCircle2 className="w-5 h-5 text-green-600" /> : <AlertCircle className="w-5 h-5 text-red-600" />}
                      </div>
                    </>
                  )}
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          <aside className="lg:col-span-1">
            <Card>
              <CardContent className="p-4">
                <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'chat' | 'kb')} className="w-full">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="chat" className="text-xs">
                      <MessageSquare className="w-4 h-4 mr-1" />
                      Chat
                    </TabsTrigger>
                    <TabsTrigger value="kb" className="text-xs">
                      <FolderOpen className="w-4 h-4 mr-1" />
                      Knowledge Base
                    </TabsTrigger>
                  </TabsList>
                </Tabs>

                {activeTab === 'chat' && (
                  <div className="mt-4 space-y-4">
                    <div>
                      <h3 className="text-sm font-semibold mb-3">Settings</h3>
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <Label htmlFor="strict-local" className="text-xs">Strict Local Mode</Label>
                          <Switch id="strict-local" checked={strictLocal} onCheckedChange={setStrictLocal} />
                        </div>
                        <p className="text-xs text-slate-500">Only use local knowledge base</p>

                        <div className="border-t pt-3">
                          <div className="flex items-center justify-between">
                            <Label htmlFor="web-search" className="text-xs">Web Search</Label>
                            <Switch id="web-search" checked={webSearchEnabled} onCheckedChange={setWebSearchEnabled} disabled={strictLocal} />
                          </div>
                          <p className="text-xs text-slate-500 mt-1">Enable web search for current info</p>
                        </div>
                      </div>
                    </div>

                    <div className="border-t pt-3">
                      <h3 className="text-sm font-semibold mb-2">Upload Documents</h3>
                      <p className="text-xs text-slate-500 mb-2">Add files or URLs to the knowledge base</p>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={() => setActiveTab('kb')} 
                        className="w-full flex items-center justify-center gap-2"
                      >
                        <Upload className="w-4 h-4" />
                        Go to Knowledge Base
                      </Button>
                    </div>

                    {messages.length > 0 && (
                      <div className="border-t pt-3">
                        <h3 className="text-sm font-semibold mb-2">Current Session</h3>
                        <p className="text-xs text-slate-600">{messages.filter(m => m.role === 'user').length} messages</p>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'kb' && (
                  <div className="mt-4 space-y-4 overflow-y-auto max-h-[500px]">
                    <div>
                      <h3 className="text-sm font-semibold mb-2">Statistics</h3>
                      {kbStats && (
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-slate-600">Documents:</span>
                            <span className="font-medium">{kbStats.document_count}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-600">Chunks:</span>
                            <span className="font-medium">{kbStats.chunk_count}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </aside>

          <div className="lg:col-span-4">
            {activeTab === 'chat' && (
              <div className="space-y-4">
                <Card className="min-h-[500px] max-h-[600px] overflow-y-auto">
                  <CardContent className="p-6 space-y-4">
                    {messages.length === 0 && (
                      <div className="text-center py-12 text-slate-500">
                        <MessageSquare className="w-12 h-12 mx-auto mb-4 text-slate-400" />
                        <p className="text-lg font-medium">Welcome to RAG System</p>
                        <p className="text-sm mt-2">Ask me anything! I can search the web, query the knowledge base, and use specialized tools.</p>
                      </div>
                    )}

                    {messages.map((message, index) => (
                      <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[85%] rounded-lg p-4 ${message.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border border-slate-200'}`}>
                          {message.role === 'user' ? (
                            <p className="whitespace-pre-wrap">{message.content}</p>
                          ) : (
                            <div className="prose prose-sm max-w-none">
                              <ReactMarkdown>{message.content}</ReactMarkdown>
                            </div>
                          )}

                          {message.role === 'assistant' && (
                            <div className="mt-3 space-y-2">
                              {message.sources_used && message.sources_used.length > 0 && (
                                <div className="flex flex-wrap gap-2">
                                  {message.sources_used.map((source, i) => (
                                    <Badge key={i} variant="outline" className={`text-xs ${getSourceColor(source)}`}>
                                      {getSourceIcon(source)}
                                      <span className="ml-1">{source.replace(/_/g, ' ')}</span>
                                    </Badge>
                                  ))}
                                </div>
                              )}

                              {message.tool_results && Object.keys(message.tool_results).length > 0 && (
                                <div>
                                  <button onClick={() => toggleDetails(index)} className="flex items-center gap-2 text-xs text-slate-600 hover:text-slate-900 transition-colors">
                                    {expandedDetails === index ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                    <span className="font-medium">View Details</span>
                                  </button>
                                  {expandedDetails === index && renderToolDetails(message.tool_results, message.sources_used || [])}
                                </div>
                              )}

                                              {message.citations && message.citations.length > 0 && (
                                                <div className="text-xs text-slate-600 space-y-1 border-t pt-2">
                                                  <p className="font-medium">Citations:</p>
                                                  {message.citations.map((citation, i) => (
                                                    <p key={i} className="pl-2 break-words overflow-wrap-anywhere">{citation}</p>
                                                  ))}
                                                </div>
                                              )}

                              <div className="flex items-center gap-3 text-xs text-slate-500 border-t pt-2">
                                {message.latency_ms && <span>{(message.latency_ms / 1000).toFixed(2)}s</span>}
                                {message.fast_path && <Badge variant="outline" className="text-xs">Fast Path</Badge>}
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

                {error && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                <Card>
                  <CardContent className="p-4">
                    <form onSubmit={handleSubmit} className="flex gap-2">
                      <Textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask me anything..." className="min-h-[60px] resize-none" onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(e) } }} />
                      <Button type="submit" disabled={loading || !input.trim()} className="self-end">
                        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                      </Button>
                    </form>
                    <p className="text-xs text-slate-500 mt-2">Press Enter to send, Shift+Enter for new line</p>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'kb' && (
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Upload className="w-5 h-5" />
                      Upload Documents
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <Label htmlFor="file-upload" className="text-sm font-medium">Select Files</Label>
                      <Input id="file-upload" type="file" multiple onChange={(e) => setUploadFiles(e.target.files)} className="mt-2" accept=".pdf,.docx,.doc,.txt,.md,.png,.jpg,.jpeg" />
                      <p className="text-xs text-slate-500 mt-1">Supported: PDF, DOCX, TXT, MD, Images</p>
                    </div>

                    <div className="border-t pt-4">
                      <Label htmlFor="url-input" className="text-sm font-medium flex items-center gap-2">
                        <LinkIcon className="w-4 h-4" />
                        Or Enter URLs
                      </Label>
                      <Input id="url-input" type="text" value={urlInput} onChange={(e) => setUrlInput(e.target.value)} placeholder="https://example.com/document.pdf (comma-separated for multiple)" className="mt-2" />
                      <p className="text-xs text-slate-500 mt-1">Enter one or more URLs separated by commas</p>
                    </div>

                    <Button onClick={handleIngest} disabled={ingesting || (!uploadFiles && !urlInput.trim())} className="w-full">
                      {ingesting ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Ingesting...
                        </>
                      ) : (
                        <>
                          <Upload className="w-4 h-4 mr-2" />
                          Ingest Documents
                        </>
                      )}
                    </Button>

                    {ingestMessage && (
                      <Alert className={ingestMessage.startsWith('Success') ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}>
                        <AlertDescription className={ingestMessage.startsWith('Success') ? 'text-green-800' : 'text-red-800'}>
                          {ingestMessage}
                        </AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Database className="w-5 h-5" />
                      Knowledge Base Status
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {kbStats ? (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                          <span className="text-sm font-medium">Total Documents</span>
                          <span className="text-2xl font-bold text-blue-600">{kbStats.document_count}</span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                          <span className="text-sm font-medium">Total Chunks</span>
                          <span className="text-2xl font-bold text-green-600">{kbStats.chunk_count}</span>
                        </div>
                        {kbStats.last_ingested_at && (
                          <div className="text-xs text-slate-600">
                            Last ingested: {new Date(kbStats.last_ingested_at).toLocaleString()}
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-sm text-slate-500">Loading statistics...</p>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App

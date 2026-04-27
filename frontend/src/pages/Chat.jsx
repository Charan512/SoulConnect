import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { Brain, ShieldAlert, Volume2, VolumeX, Mic, Square, Trash2 } from 'lucide-react';

const SUGGESTIONS = [
    "I've been feeling really anxious lately",
    "I need someone to talk to",
    "Help me calm down",
];

export default function Chat() {
    const { user, logout, authFetch } = useAuth();

    // Auto-logout helper — called whenever backend returns 401
    const handleUnauthorized = () => {
        logout();
    };
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [sessions, setSessions] = useState([]);
    const [currentSessionId, setCurrentSessionId] = useState('default');
    const [speakingId, setSpeakingId] = useState(null);
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const messagesEnd = useRef(null);

    // Stop speaking when component unmounts
    useEffect(() => {
        return () => {
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
            if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                mediaRecorderRef.current.stop();
            }
        };
    }, []);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                const reader = new FileReader();

                reader.readAsDataURL(audioBlob);
                reader.onloadend = async () => {
                    const base64Audio = reader.result.split(',')[1];
                    try {
                        setLoading(true);
                        const res = await authFetch('/speech-to-text', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                audio_base64: base64Audio,
                                source_language: "en"
                            })
                        });
                        const data = await res.json();
                        if (res.ok && data.text) {
                            setInput(prev => prev ? prev + ' ' + data.text : data.text);
                        } else {
                            console.error("Transcribe Error:", data);
                        }
                    } catch (e) {
                        console.error("Audio API fetch failed:", e);
                    } finally {
                        setLoading(false);
                    }
                };
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error("Microphone access denied or not supported:", error);
            alert("Could not access microphone.");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    useEffect(() => {
        // Load sessions and then load the history for the most recent one
        const initChat = async () => {
            try {
                const res = await authFetch('/sessions');
                if (res.ok) {
                    const data = await res.json();
                    setSessions(data);

                    // If there are sessions, load the most recent one. Otherwise, load 'default'
                    if (data && data.length > 0) {
                        loadSessionHistory(data[0].session_id);
                    } else {
                        loadSessionHistory('default');
                    }
                }
            } catch (err) {
                console.error("Failed to load sessions:", err);
                loadSessionHistory('default');
            }
        };

        initChat();
    }, []);

    useEffect(() => {
        messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    const loadSessions = async () => {
        try {
            const res = await authFetch('/sessions');
            if (res.status === 401) { handleUnauthorized(); return; }
            if (res.ok) {
                const data = await res.json();
                setSessions(data);
            }
        } catch { }
    };

    const loadSessionHistory = async (sessionId) => {
        setCurrentSessionId(sessionId);
        setLoading(true);
        try {
            const res = await authFetch(`/history/${sessionId}?limit=50`);
            if (res.status === 401) { handleUnauthorized(); return; }
            if (res.ok) {
                const data = await res.json();
                const formatted = [];
                // Data comes back in DESC order (latest first), so we reverse to display chronically
                data.reverse().forEach(record => {
                    formatted.push({ role: 'user', text: record.user_msg });
                    formatted.push({
                        role: 'bot',
                        text: record.bot_msg,
                        meta: { risk: record.risk, sentiment: record.sentiment }
                    });
                });
                setMessages(formatted);
            }
        } catch { }
        setLoading(false);
    };

    const startNewChat = () => {
        setCurrentSessionId(Date.now().toString());
        setMessages([]);
    };

    const deleteSession = async (e, sessionId) => {
        e.stopPropagation(); // prevent clicking the history item
        try {
            const res = await authFetch(`/history/${sessionId}`, { method: 'DELETE' });
            if (res.ok) {
                setSessions(prev => prev.filter(s => s.session_id !== sessionId));
                if (currentSessionId === sessionId) {
                    startNewChat();
                }
            }
        } catch (err) {
            console.error("Failed to delete session", err);
        }
    };

    const sendMessage = async (e, overrideText) => {
        if (e) e.preventDefault();
        const userText = (overrideText || input).trim();
        if (!userText || loading) return;

        setInput('');
        setMessages(prev => [...prev, { role: 'user', text: userText }]);
        setLoading(true);

        try {
            const res = await authFetch('/chat', {
                method: 'POST',
                body: JSON.stringify({ text: userText, session_id: currentSessionId }),
            });

            if (res.status === 401) {
                // JWT expired or invalid — log the user out immediately
                handleUnauthorized();
                return;
            }

            if (res.ok) {
                const data = await res.json();
                setMessages(prev => [...prev, { role: 'bot', text: data.response, meta: data }]);

                // Refresh sessions silently in background so sidebar updates if this was first message
                if (messages.length === 0) {
                    loadSessions();
                }
            } else {
                const err = await res.json();
                setMessages(prev => [...prev, {
                    role: 'bot',
                    text: err.detail || 'Something went wrong. Please try again.',
                    meta: null,
                }]);
            }
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'bot',
                text: 'Connection error. Please check if the backend is running.',
                meta: null,
            }]);
        }

        setLoading(false);
    };

    const handleChip = (text) => {
        sendMessage(null, text);
    };

    const handleSpeak = (text, index) => {
        if (!window.speechSynthesis) {
            alert("Sorry, your browser doesn't support text to speech!");
            return;
        }

        // Toggle off if clicking the same speaking message
        if (speakingId === index) {
            window.speechSynthesis.cancel();
            setSpeakingId(null);
            return;
        }

        // Cancel any currently playing speech
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);

        // Find a feminine voice
        const voices = window.speechSynthesis.getVoices();
        const feminineVoices = voices.filter(v =>
            v.name.includes("Female") ||
            v.name.includes("female") ||
            v.name.includes("Samantha") ||
            v.name.includes("Victoria") ||
            v.name.includes("Karen") ||
            v.name.includes("Tessa") ||
            v.name.includes("Monica") ||
            v.name.includes("Moira") ||
            v.name.includes("Luciana") ||
            v.name.includes("Amelie") ||
            v.name.includes("Kyoko") ||
            v.name.includes("Siri Female") ||
            v.name.includes("Google UK English Female") ||
            v.name.includes("Google US English") ||
            (v.name.includes("English") && v.name.includes("United States") && v.gender === "female") ||
            (v.name.includes("English") && v.name.includes("UK") && v.gender === "female")
        );

        // Fallback to the first feminine voice, or default if none exist
        if (feminineVoices.length > 0) {
            utterance.voice = feminineVoices[0];
        }

        utterance.rate = 0.95; // Slightly slower feels more calming/empathetic

        utterance.onend = () => setSpeakingId(null);
        utterance.onerror = () => setSpeakingId(null);

        setSpeakingId(index);
        window.speechSynthesis.speak(utterance);
    };

    return (
        <div className="chat-layout">
            {/* ── Sidebar ── */}
            <div className="sidebar">
                <div className="sidebar-header">
                    <div className="sidebar-brand">
                        <div className="sidebar-brand-icon">
                            <Brain size={16} color="#fff" />
                        </div>
                        <h2>Soul Connect</h2>
                    </div>
                    <p>Your safe space</p>
                </div>

                <div style={{ padding: '16px 16px 0' }}>
                    <button className="btn-new-chat" onClick={startNewChat}>
                        <span style={{ fontSize: '18px', lineHeight: 1 }}>+</span>
                        <span>New Chat</span>
                    </button>
                </div>

                <div className="sidebar-section-label">
                    <span>Past Chats</span>
                </div>

                <div className="sidebar-history">
                    {sessions.length === 0 && (
                        <p style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '8px 12px', textAlign: 'center' }}>
                            No conversations yet
                        </p>
                    )}
                    {sessions.map((s) => (
                        <div
                            className={`history-item ${currentSessionId === s.session_id ? 'active' : ''}`}
                            key={s.session_id}
                            onClick={() => loadSessionHistory(s.session_id)}
                            style={currentSessionId === s.session_id ? { background: 'var(--blue-soft)', borderColor: 'var(--blue-light)', fontWeight: '500' } : {}}
                        >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                                <span className="risk-dot risk-low"></span>
                                <span style={{ textOverflow: 'ellipsis', whiteSpace: 'nowrap', overflow: 'hidden' }}>{s.title}</span>
                            </div>
                            <button 
                                className="btn-delete-session"
                                onClick={(e) => deleteSession(e, s.session_id)}
                                title="Delete chat"
                            >
                                <Trash2 size={13} />
                            </button>
                        </div>
                    ))}
                </div>

                <div className="sidebar-footer">
                    <div className="user-info">
                        <div className="user-avatar">
                            {user?.username?.[0]?.toUpperCase() || '?'}
                        </div>
                        <span className="user-name">{user?.username || 'User'}</span>
                    </div>
                    <button className="btn-logout" onClick={logout}>Sign Out</button>
                </div>
            </div>

            {/* ── Chat Area ── */}
            <div className="chat-area">

                {/* Header bar */}
                <div className="chat-header">
                    <div className="chat-header-icon">
                        <Brain size={18} color="#fff" />
                    </div>
                    <div className="chat-header-text">
                        <h2>Soul Connect</h2>
                        <p>AI Mental Health Companion</p>
                    </div>
                    <div className="status-dot" title="Online" />
                </div>

                {/* Messages */}
                <div className="chat-messages">
                    {messages.length === 0 && (
                        <div className="empty-state">
                            <div className="empty-icon">
                                <Brain size={28} color="var(--violet-light)" />
                            </div>
                            <h3>How are you feeling today?</h3>
                            <p>
                                Share what's on your mind. I'm here to listen, support,
                                and offer gentle guidance — everything is private.
                            </p>
                            <div className="suggestion-chips">
                                {SUGGESTIONS.map((s) => (
                                    <button
                                        key={s}
                                        className="chip"
                                        onClick={() => handleChip(s)}
                                    >
                                        {s}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <React.Fragment key={i}>
                            <div className={`message-row ${msg.role}`}>
                                {msg.role === 'bot' && (
                                    <div className="bot-avatar">
                                        <Brain size={13} color="#fff" />
                                    </div>
                                )}
                                <div className="message-bubble">
                                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</div>
                                    {msg.role === 'bot' && (
                                        <div className="message-actions">
                                            <button
                                                className={`btn-speak ${speakingId === i ? 'active' : ''}`}
                                                onClick={() => handleSpeak(msg.text, i)}
                                                title={speakingId === i ? "Stop speaking" : "Listen aloud"}
                                            >
                                                {speakingId === i ? <VolumeX size={14} /> : <Volume2 size={14} />}
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>


                        </React.Fragment>
                    ))}

                    {loading && (
                        <div className="message-row bot">
                            <div className="bot-avatar">
                                <Brain size={13} color="#fff" />
                            </div>
                            <div className="message-bubble">
                                <div className="typing-indicator">
                                    <span></span><span></span><span></span>
                                </div>
                            </div>
                        </div>
                    )}

                    <div ref={messagesEnd} />
                </div>

                {/* Input */}
                <div className="chat-input-area">
                    <form className="chat-input-wrapper" onSubmit={sendMessage}>
                        <input
                            type="text"
                            placeholder="Share what's on your mind..."
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            disabled={loading}
                            autoFocus
                        />
                        <button 
                            type="button" 
                            style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: isRecording ? '#ef4444' : '#6b7280', margin: '0 8px' }}
                            onClick={isRecording ? stopRecording : startRecording}
                            disabled={loading}
                            title={isRecording ? "Stop Recording" : "Use Microphone"}
                        >
                            {isRecording ? <Square size={20} fill="currentColor" /> : <Mic size={20} />}
                        </button>
                        <button type="submit" className="btn-send" disabled={loading || !input.trim()}>
                            ↑
                        </button>
                    </form>
                </div>

            </div>
        </div>
    );
}

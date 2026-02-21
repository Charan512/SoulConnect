import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { Brain, ShieldAlert } from 'lucide-react';

export default function Chat() {
    const { user, logout, authFetch } = useAuth();
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [history, setHistory] = useState([]);
    const messagesEnd = useRef(null);

    useEffect(() => {
        loadHistory();
    }, []);

    useEffect(() => {
        messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    const loadHistory = async () => {
        try {
            const res = await authFetch('/history?limit=30');
            if (res.ok) {
                const data = await res.json();
                setHistory(data.reverse());
            }
        } catch { }
    };

    const sendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const userText = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', text: userText }]);
        setLoading(true);

        try {
            const res = await authFetch('/chat', {
                method: 'POST',
                body: JSON.stringify({ text: userText, session_id: 'default' }),
            });

            if (res.ok) {
                const data = await res.json();
                setMessages(prev => [...prev, { role: 'bot', text: data.response, meta: data }]);

                setHistory(prev => [...prev, {
                    id: Date.now(),
                    user_msg: userText,
                    bot_msg: data.response,
                    risk: data.risk,
                    sentiment: data.sentiment,
                    time: new Date().toISOString(),
                }]);
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

    const getRiskClass = (risk) => {
        if (!risk) return 'risk-low';
        return risk === 'HIGH' ? 'risk-high' : risk === 'MEDIUM' ? 'risk-medium' : 'risk-low';
    };

    return (
        <div className="chat-layout">
            {/* Sidebar */}
            <div className="sidebar">
                <div className="sidebar-header">
                    <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Brain size={20} color="#1e40af" />
                        Soul Connect
                    </h2>
                    <p>Your safe space</p>
                </div>

                <div className="sidebar-history">
                    {history.length === 0 && (
                        <p style={{ color: '#64748b', fontSize: '13px', padding: '12px', textAlign: 'center' }}>
                            No conversations yet
                        </p>
                    )}
                    {history.slice(-20).reverse().map((h, i) => (
                        <div className="history-item" key={h.id || i}>
                            <span className={`risk-dot ${getRiskClass(h.risk)}`}></span>
                            {h.user_msg}
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

            {/* Chat Area */}
            <div className="chat-area">
                <div className="chat-messages">
                    {messages.length === 0 && (
                        <div className="empty-state">
                            <h3>How are you feeling today?</h3>
                            <p>
                                Share what's on your mind. I'm here to listen, support,
                                and offer gentle guidance. Everything shared here is private.
                            </p>
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <React.Fragment key={i}>
                            <div className={`message-row ${msg.role}`}>
                                <div className="message-bubble">
                                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</div>

                                </div>
                            </div>

                            {msg.role === 'bot' && msg.meta?.emergency_contact && (
                                <div className="emergency-banner" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <ShieldAlert size={18} />
                                    <span>Emergency alert triggered — contact notified: <strong>{msg.meta.emergency_contact}</strong></span>
                                </div>
                            )}
                        </React.Fragment>
                    ))}

                    {loading && (
                        <div className="message-row bot">
                            <div className="message-bubble">
                                <div className="typing-indicator">
                                    <span></span><span></span><span></span>
                                </div>
                            </div>
                        </div>
                    )}

                    <div ref={messagesEnd} />
                </div>

                <div className="chat-input-area">
                    <form className="chat-input-wrapper" onSubmit={sendMessage}>
                        <input
                            type="text"
                            placeholder="Type your message..."
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            disabled={loading}
                            autoFocus
                        />
                        <button type="submit" className="btn-send" disabled={loading || !input.trim()}>
                            ↑
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}

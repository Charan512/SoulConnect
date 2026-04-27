import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Brain, Sparkles, Shield, Lock } from 'lucide-react';

export default function Login() {
    const { login } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await login(username, password);
        } catch (err) {
            setError(err.message);
        }
        setLoading(false);
    };

    return (
        <div className="auth-container">
            <div className="auth-wrapper">

                {/* ── Left brand panel ── */}
                <div className="auth-brand">
                    <div className="auth-brand-logo">
                        <div className="logo-icon">
                            <Brain size={24} color="#fff" />
                        </div>
                        <h1>Soul Connect</h1>
                    </div>

                    <p className="auth-brand-tagline">
                        Your mind deserves a <span>safe place</span> to breathe.
                    </p>
                    <p className="auth-brand-sub">
                        An AI companion that listens without judgment — here for you, anytime.
                    </p>

                    <div className="auth-brand-badges">
                        <div className="auth-badge">
                            <div className="auth-badge-dot" />
                            End-to-end encrypted conversations
                        </div>
                        <div className="auth-badge">
                            <div className="auth-badge-dot" />
                            AI trained in compassionate support
                        </div>
                        <div className="auth-badge">
                            <div className="auth-badge-dot" />
                            Emergency contact system included
                        </div>
                    </div>
                </div>

                {/* ── Right form panel ── */}
                <div className="auth-form-panel">
                    <p className="auth-form-title">Welcome back</p>
                    <p className="auth-form-subtitle">Sign in to continue your journey</p>

                    {error && <div className="error-message">{error}</div>}

                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label>Username</label>
                            <input
                                type="text"
                                placeholder="Enter your username"
                                value={username}
                                onChange={e => setUsername(e.target.value)}
                                required
                                autoFocus
                            />
                        </div>
                        <div className="form-group">
                            <label>Password</label>
                            <input
                                type="password"
                                placeholder="Enter your password"
                                value={password}
                                onChange={e => setPassword(e.target.value)}
                                required
                            />
                        </div>
                        <button type="submit" className="btn-primary" disabled={loading}>
                            {loading ? 'Signing in...' : 'Sign In'}
                        </button>
                    </form>

                    <div className="auth-switch">
                        Don't have an account? <Link to="/register">Create one</Link>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', marginTop: '20px', fontSize: '11px', color: 'var(--text-muted)' }}>
                        <Lock size={11} />
                        Secure &amp; Private — your data stays yours
                    </div>
                </div>

            </div>
        </div>
    );
}

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Brain, ShieldAlert, Lock } from 'lucide-react';

export default function Register() {
    const { register } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [emergencyContact, setEmergencyContact] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await register(username, password, emergencyContact);
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
                        Take the first step toward <span>healing</span>.
                    </p>
                    <p className="auth-brand-sub">
                        Create your account and start talking to an AI that genuinely cares about your well-being.
                    </p>

                    <div className="auth-brand-badges">
                        <div className="auth-badge">
                            <div className="auth-badge-dot" />
                            Private, encrypted conversations
                        </div>
                        <div className="auth-badge">
                            <div className="auth-badge-dot" />
                            Real-time emotional risk detection
                        </div>
                        <div className="auth-badge">
                            <div className="auth-badge-dot" />
                            Emergency contact safety net
                        </div>
                    </div>
                </div>

                {/* ── Right form panel ── */}
                <div className="auth-form-panel">
                    <p className="auth-form-title">Create your account</p>
                    <p className="auth-form-subtitle">Join Soul Connect — it's free</p>

                    {error && <div className="error-message">{error}</div>}

                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label>Username</label>
                            <input
                                type="text"
                                placeholder="Choose a username"
                                value={username}
                                onChange={e => setUsername(e.target.value)}
                                required
                                autoFocus
                                minLength={3}
                            />
                        </div>
                        <div className="form-group">
                            <label>Password</label>
                            <input
                                type="password"
                                placeholder="Create a password (min 4 chars)"
                                value={password}
                                onChange={e => setPassword(e.target.value)}
                                required
                                minLength={4}
                            />
                        </div>
                        <div className="form-group">
                            <label>Emergency Contact Number</label>
                            <input
                                type="tel"
                                placeholder="e.g. +91 98204 66726"
                                value={emergencyContact}
                                onChange={e => setEmergencyContact(e.target.value)}
                                required
                                minLength={5}
                            />
                        </div>

                        <div className="privacy-note">
                            <ShieldAlert size={14} style={{ marginTop: '1px', flexShrink: 0, color: 'var(--violet-light)' }} />
                            This number is only contacted if our AI detects you may be in crisis. It is never shared.
                        </div>

                        <button type="submit" className="btn-primary" disabled={loading}>
                            {loading ? 'Creating account...' : 'Create Account'}
                        </button>
                    </form>

                    <div className="auth-switch">
                        Already have an account? <Link to="/login">Sign in</Link>
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

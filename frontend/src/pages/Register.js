import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Brain, ShieldAlert } from 'lucide-react';

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
            <div className="auth-card">
                <div className="auth-logo">
                    <h1 style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                        <Brain size={28} color="#1e40af" />
                        Soul Connect
                    </h1>
                    <p>Create your account</p>
                </div>

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
                            placeholder="Create a password"
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
                    <p style={{ display: 'flex', alignItems: 'flex-start', gap: '6px', fontSize: '12px', color: '#64748b', marginBottom: '16px', lineHeight: '1.5' }}>
                        <ShieldAlert size={14} style={{ marginTop: '2px', flexShrink: 0 }} />
                        This number will be contacted if our system detects you may be in crisis.
                    </p>
                    <button type="submit" className="btn-primary" disabled={loading}>
                        {loading ? 'Creating account...' : 'Create Account'}
                    </button>
                </form>

                <div className="auth-switch">
                    Already have an account? <Link to="/login">Sign in</Link>
                </div>
            </div>
        </div>
    );
}

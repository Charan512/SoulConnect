import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

// Use Vercel Environment Variable if available, otherwise default to Vite local proxy
const API = import.meta.env.VITE_API_URL || '/api';

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(localStorage.getItem('token'));
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (token) {
            fetch(`${API}/me`, { headers: { Authorization: `Bearer ${token}`, 'ngrok-skip-browser-warning': 'true' } })
                .then(r => {
                    if (!r.ok) throw new Error('Invalid token');
                    return r.json();
                })
                .then(data => { setUser(data); setLoading(false); })
                .catch(() => { logout(); setLoading(false); });
        } else {
            setLoading(false);
        }
    }, [token]);

    const login = async (username, password) => {
        const res = await fetch(`${API}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
            body: JSON.stringify({ username, password }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Login failed');
        }
        const data = await res.json();
        localStorage.setItem('token', data.token);
        setToken(data.token);
        setUser({ id: data.user_id, username: data.username });
        return data;
    };

    const register = async (username, password, emergency_contact) => {
        const res = await fetch(`${API}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
            body: JSON.stringify({ username, password, emergency_contact }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Registration failed');
        }
        const data = await res.json();
        localStorage.setItem('token', data.token);
        setToken(data.token);
        setUser({ id: data.user_id, username: data.username });
        return data;
    };

    const logout = () => {
        localStorage.removeItem('token');
        setToken(null);
        setUser(null);
    };

    const authFetch = (url, options = {}) => {
        return fetch(`${API}${url}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true',
                Authorization: `Bearer ${token}`,
                ...options.headers,
            },
        });
    };

    return (
        <AuthContext.Provider value={{ user, token, login, register, logout, loading, authFetch }}>
            {children}
        </AuthContext.Provider>
    );
}

export const useAuth = () => useContext(AuthContext);

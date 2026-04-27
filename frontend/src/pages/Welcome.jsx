import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Heart, Sparkles } from 'lucide-react';

export default function Welcome() {
    const navigate = useNavigate();

    return (
        <div className="welcome-container">
            <div className="welcome-content">
                <div className="welcome-icon-box">
                    <Heart className="welcome-icon heartbeat" />
                </div>

                <h1 className="welcome-title">
                    Welcome to <span>Soul Connect</span>
                </h1>

                <p className="welcome-subtitle">
                    Your personal sanctuary for mental well-being. We're here to listen, support, and guide you through whatever you're feeling today.
                </p>

                <div className="welcome-actions">
                    <button
                        className="btn-primary welcome-btn"
                        onClick={() => navigate('/landing')}
                    >
                        <span>Let's Go</span>
                        <Sparkles className="btn-icon" size={18} />
                    </button>
                </div>
            </div>

            {/* Decorative background elements matching the auth pages */}
            <div className="decor-orb orb-1"></div>
            <div className="decor-orb orb-2"></div>
        </div>
    );
}

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain, HeartHandshake, ShieldCheck, Activity, ArrowRight, Sparkles, ChevronDown, MessageCircle, Mail } from 'lucide-react';

export default function Landing() {
    const navigate = useNavigate();
    const [openFaq, setOpenFaq] = useState(null);

    const toggleFaq = (index) => {
        setOpenFaq(openFaq === index ? null : index);
    };

    const faqs = [
        { q: "Is Soul Connect a substitute for therapy?", a: "No. While Soul Connect provides empathetic listening and CBT-based exercises, it is not a replacement for professional psychological treatment. In a severe crisis, please contact emergency services." },
        { q: "Is my data private and secure?", a: "Yes. Your conversations are securely stored and tied to your account for session continuity, but they remain completely private." },
        { q: "How does the Emergency Support work?", a: "If our intelligent models detect high-risk language or severe distress, Soul Connect can automatically alert your designated emergency contact, ensuring someone is there for you." },
        { q: "Which AI model do you use?", a: "We use a customized, fast, empathetic conversational AI fine-tuned specifically for emotional support and therapeutic conversations." }
    ];

    return (
        <div className="landing-page">
            <nav className="landing-navbar">
                <div className="nav-brand">
                    <div className="nav-logo">
                        <HeartHandshake className="text-white" size={24} />
                    </div>
                    <span>Soul Connect</span>
                </div>
                <div className="nav-actions">
                    <button onClick={() => navigate('/login')} className="btn-text">Log In</button>
                    <button onClick={() => navigate('/register')} className="btn-solid">Sign Up</button>
                </div>
            </nav>

            <main className="landing-main">
                <section className="hero-section">
                    <div className="hero-content">
                        <div className="badge-pill">
                            <Sparkles size={14} className="text-blue" />
                            <span>Your AI Mental Health Companion</span>
                        </div>
                        <h1 className="hero-title">
                            A safe space for your mind to <span>breathe and heal</span>.
                        </h1>
                        <p className="hero-subtitle">
                            Empathetic conversations, personalized therapeutic activities, and intelligent risk analysis. Whenever you need it, wherever you are.
                        </p>
                        <button
                            className="btn-primary hero-btn"
                            onClick={() => navigate('/register')}
                        >
                            Start Your Journey <ArrowRight size={18} />
                        </button>
                    </div>

                    {/* Abstract Hero Image/Illustration */}
                    <div className="hero-visual">
                        <div className="glass-card visual-card-main">
                            <div className="visual-header">
                                <div className="bot-avatar"><Brain size={20} /></div>
                                <div>
                                    <h4>AI Companion</h4>
                                    <p>Online</p>
                                </div>
                            </div>
                            <div className="visual-body">
                                <div className="chat-bubble bot-bubble">I hear you, and your feelings matter. I'm here to support you.</div>
                                <div className="chat-bubble user-bubble">Thank you. I'm feeling a bit overwhelmed today.</div>
                                <div className="chat-bubble bot-bubble">It's okay to feel that way. Let's try a quick grounding exercise together.</div>
                            </div>
                        </div>
                        <div className="glass-card visual-card-floating">
                            <ShieldCheck size={24} className="text-emerald" />
                            <span>100% Secure & Private</span>
                        </div>
                    </div>
                </section>

                <section className="features-section">
                    <div className="section-head">
                        <h2>Thoughtful features for your well-being</h2>
                        <p>Designed with care to provide meaningful support across different emotional states.</p>
                    </div>

                    <div className="features-grid">
                        <div className="feature-card">
                            <div className="feature-icon bg-blue-soft"><Brain className="text-blue" size={28} /></div>
                            <h3>Cognitive Behavioral Therapy</h3>
                            <p>Gentle guidance to help you reframe negative thoughts and build healthier mental habits.</p>
                        </div>
                        <div className="feature-card">
                            <div className="feature-icon bg-emerald-soft"><Activity className="text-emerald" size={28} /></div>
                            <h3>Risk & Emotion Analysis</h3>
                            <p>Intelligent understanding of your emotional state to provide the most appropriate support mode.</p>
                        </div>
                        <div className="feature-card">
                            <div className="feature-icon bg-amber-soft"><HeartHandshake className="text-amber" size={28} /></div>
                            <h3>Emergency Support</h3>
                            <p>Immediate connections and alerts to your designated emergency contacts when you need them most.</p>
                        </div>
                    </div>
                </section>

                <section className="about-ai-section">
                    <div className="about-content">
                        <h2>Powered by Empathetic AI</h2>
                        <p>Our intelligent system doesn't just read your words—it understands your emotional state. Using advanced natural language processing, Soul Connect identifies the underlying feelings and seamlessly adapts its therapeutic approach to what you need.</p>
                        <ul className="about-points">
                            <li><Sparkles size={18} className="text-blue" /> Sentiment tracking across sessions</li>
                            <li><Activity size={18} className="text-emerald" /> Real-time risk assessment</li>
                            <li><Brain size={18} className="text-amber" /> Context-aware CBT exercises</li>
                        </ul>
                    </div>
                    <div className="about-image-wrapper">
                        <img src="/images/ai_therapy.png" alt="AI Conceptual Mind" className="about-img" />
                    </div>
                </section>

                <section className="community-section">
                    <div className="community-image-wrapper">
                        <img src="/images/community.png" alt="Supportive Community" className="community-img" />
                    </div>
                    <div className="community-content">
                        <h2>You Are Not Alone</h2>
                        <p>Mental health is a journey, and having a safe space to express yourself is the first step towards healing. Whether you are dealing with daily stress, anxiety, or simply need an outlet, our platform is designed to offer a comforting presence.</p>
                        <button className="btn-secondary" onClick={() => navigate('/register')}>Join Our Safe Space</button>
                    </div>
                </section>

                <section className="faq-section">
                    <div className="section-head">
                        <h2>Frequently Asked Questions</h2>
                        <p>Everything you need to know about Soul Connect.</p>
                    </div>
                    <div className="faq-list">
                        {faqs.map((faq, idx) => (
                            <div key={idx} className={`faq-item ${openFaq === idx ? 'open' : ''}`} onClick={() => toggleFaq(idx)}>
                                <div className="faq-question">
                                    <h3>{faq.q}</h3>
                                    <ChevronDown size={20} className="faq-icon" />
                                </div>
                                <div className="faq-answer">
                                    <p>{faq.a}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                <section className="contact-section">
                    <div className="contact-container">
                        <div className="contact-info">
                            <h2>Get in Touch</h2>
                            <p>Have questions or feedback? We'd love to hear from you. Our team is dedicated to improving the support you receive.</p>
                            <div className="contact-methods">
                                <div className="contact-method">
                                    <div className="contact-icon bg-blue-soft"><Mail size={20} className="text-blue" /></div>
                                    <span>support@soulconnect.ai</span>
                                </div>
                                <div className="contact-method">
                                    <div className="contact-icon bg-emerald-soft"><MessageCircle size={20} className="text-emerald" /></div>
                                    <span>Live Chat (Coming Soon)</span>
                                </div>
                            </div>
                        </div>
                        <div className="contact-form">
                            <input type="text" placeholder="Your Name" className="form-input" />
                            <input type="email" placeholder="Your Email" className="form-input" />
                            <textarea placeholder="How can we help?" className="form-textarea" rows="4"></textarea>
                            <button className="btn-primary" style={{ width: '100%', marginTop: '8px' }}>Send Message</button>
                        </div>
                    </div>
                </section>
            </main>

            <footer className="landing-footer">
                <div className="footer-content">
                    <div className="footer-brand">
                        <div className="nav-logo mb-2" style={{ margin: '0 auto 12px' }}>
                            <HeartHandshake className="text-white" size={20} />
                        </div>
                        <span>Soul Connect &copy; {new Date().getFullYear()}</span>
                        <p>Not a substitute for professional medical advice or treatment.</p>
                    </div>
                </div>
            </footer>
        </div>
    );
}

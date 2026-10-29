"use client";

import { useState } from "react";

type LoginRole = "user" | "admin" | null;

export default function Home() {
  const [role, setRole] = useState<LoginRole>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");

  function openLogin(selectedRole: Exclude<LoginRole, null>) {
    setRole(selectedRole);
    setEmail("");
    setPassword("");
    setMessage("");
  }

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!email.trim() || !password) {
      setMessage("Please enter your email and password.");
      return;
    }
    setMessage("Interface ready. Connect an authentication backend to sign in.");
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <a href="/" className="brand" aria-label="GhostDemandAI home"
          onClick={() => setRole(null)}>
          <span className="brand-icon">
            <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
              <path d="M8 20C8 10.6 15.2 4 24 4s16 6.6 16 16v20l-8-5-8 5-8-5-8 5V20Z" fill="currentColor" />
              <circle cx="18" cy="19" r="3" fill="white" />
              <circle cx="30" cy="19" r="3" fill="white" />
            </svg>
          </span>
          <span>GhostDemand<span className="brand-accent">AI</span></span>
        </a>
        <span className="topbar-note">Smarter Decisions. Powered by AI.</span>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow"><span className="status-dot" /> INTELLIGENT DEMAND ANALYTICS</span>
          <h1>Smarter Demand.<br />Greater <span>Possibilities.</span></h1>
          <p className="hero-description">
            GhostDemandAI uses the power of artificial intelligence to help you
            predict, manage, and meet demand — efficiently, accurately, and at scale.
          </p>

          <div className="features">
            <article className="feature">
              <div className="feature-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none"><path d="M3 20V13M9 20V9M15 20V5M21 20V2M2 8l6-4 5 3 8-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </div>
              <h3>Data-Driven</h3><p>Actionable insights</p>
            </article>
            <article className="feature">
              <div className="feature-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none"><path d="m13 2-9 12h7l-1 8 10-13h-7l0-7Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </div>
              <h3>Smart Forecasts</h3><p>AI-powered predictions</p>
            </article>
            <article className="feature">
              <div className="feature-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" /><circle cx="12" cy="12" r="5" stroke="currentColor" strokeWidth="2" /><circle cx="12" cy="12" r="1.5" fill="currentColor" /><path d="m14 10 7-7m-5 0h5v5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </div>
              <h3>Better Decisions</h3><p>Insights you can use</p>
            </article>
          </div>

          <div className="analytics-visual" aria-label="Illustration of an AI analytics dashboard">
            <div className="visual-glow" />
            <div className="dashboard-card">
              <div className="dashboard-top">
                <span className="dashboard-dot" /><span className="dashboard-dot" /><span className="dashboard-dot" />
                <span className="dashboard-label">Demand Overview</span>
              </div>
              <div className="dashboard-body">
                <div className="metric"><span>Demand forecast</span><strong>AI Insights</strong></div>
                <div className="chart">
                  <div className="chart-grid" />
                  <svg viewBox="0 0 320 115" preserveAspectRatio="none" className="chart-line" aria-hidden="true">
                    <defs><linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2782ff" stopOpacity=".28" /><stop offset="100%" stopColor="#2782ff" stopOpacity="0" /></linearGradient></defs>
                    <path d="M0 95 L45 72 L85 82 L130 40 L175 55 L220 22 L265 35 L320 7 L320 115 L0 115 Z" fill="url(#chartFill)" />
                    <path d="M0 95 L45 72 L85 82 L130 40 L175 55 L220 22 L265 35 L320 7" fill="none" stroke="#2782ff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                    {[45, 130, 220, 320].map((x, i) => <circle key={x} cx={x} cy={[72, 40, 22, 7][i]} r="4" fill="#fff" stroke="#2782ff" strokeWidth="3" />)}
                  </svg>
                </div>
                <div className="chart-footer"><span>Historical trends</span><span className="trend">↗ Forecasting</span></div>
              </div>
            </div>
            <div className="floating-card floating-card-top"><span className="floating-symbol">✦</span><div><strong>AI Powered</strong><span>Intelligent insights</span></div></div>
            <div className="floating-card floating-card-bottom"><span className="floating-symbol">↗</span><div><strong>Plan with confidence</strong><span>Make informed decisions</span></div></div>
          </div>
        </div>

        <div className="login-area">
          {role === null ? (
            <>
              <div className="login-heading">
                <span className="heading-icon">✦</span>
                <h2>Welcome to GhostDemandAI</h2>
                <p>Choose your login type to continue</p>
              </div>
              <div className="login-cards">
                <article className="login-card user-card">
                  <div className="login-icon user-icon">
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="7" r="4" fill="currentColor" /><path d="M4 21v-2a8 8 0 0 1 16 0v2H4Z" fill="currentColor" /></svg>
                  </div>
                  <h3>User Login</h3>
                  <p>Access your account, explore demand insights, view forecasts, and manage your data.</p>
                  <button className="login-button" onClick={() => openLogin("user")}>Login as User <span>→</span></button>
                  <span className="card-caption">Your insights, all in one place</span>
                </article>
                <article className="login-card admin-card">
                  <div className="login-icon admin-icon">
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 2 21 6v6c0 5-3.8 8.5-9 10-5.2-1.5-9-5-9-10V6l9-4Z" fill="currentColor" /><path d="m8 12 2.5 2.5L16 9" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                  </div>
                  <h3>Admin Login</h3>
                  <p>Manage users, oversee operations, monitor the platform, and configure system settings.</p>
                  <button className="login-button" onClick={() => openLogin("admin")}>Login as Admin <span>→</span></button>
                  <span className="card-caption">Secure administrative access</span>
                </article>
              </div>
              <div className="security-note">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="5" y="10" width="14" height="11" rx="2" stroke="currentColor" strokeWidth="1.8" /><path d="M8 10V7a4 4 0 0 1 8 0v3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
                Secure access for every role
              </div>
            </>
          ) : (
            <div className="auth-panel">
              <button className="back-button" onClick={() => { setRole(null); setMessage(""); }}>← Back to home</button>
              <div className={`login-icon ${role === "user" ? "user-icon" : "admin-icon"}`}>{role === "user" ? "♙" : "✦"}</div>
              <h2>{role === "user" ? "User Login" : "Admin Login"}</h2>
              <p className="auth-subtitle">{role === "user" ? "Sign in to explore your demand insights." : "Sign in to access the administration portal."}</p>
              <form onSubmit={handleSubmit} className="auth-form">
                <label htmlFor="email">Email address</label>
                <input id="email" type="email" autoComplete="username" placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} required />
                <label htmlFor="password">Password</label>
                <input id="password" type="password" autoComplete="current-password" placeholder="Enter your password" value={password} onChange={(e) => setPassword(e.target.value)} required />
                <button type="submit" className="login-button">Sign in as {role === "user" ? "User" : "Admin"} <span>→</span></button>
                {message && <p className="form-message" role="status">{message}</p>}
              </form>
              {role === "admin" && <p className="admin-warning">Administrator access should be restricted to authorized accounts.</p>}
            </div>
          )}
        </div>
      </section>

      <footer className="footer">
        <span>© {new Date().getFullYear()} GhostDemandAI</span>
        <span>Intelligence for better decisions.</span>
      </footer>
    </main>
  );
}

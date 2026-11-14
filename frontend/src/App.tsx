
import { useState } from 'react'
import './App.css'

type Section = 'Workspace' | 'Transparency' | 'Memory' | 'Privacy' | 'Settings'

const navigation: { name: Section; icon: string }[] = [
  { name: 'Workspace', icon: '◈' },
  { name: 'Transparency', icon: '◎' },
  { name: 'Memory', icon: '◫' },
  { name: 'Privacy', icon: '◇' },
  { name: 'Settings', icon: '⚙' },
]

function App() {
  const [activeSection, setActiveSection] = useState<Section>('Workspace')
  const [question, setQuestion] = useState('')

  const renderWorkspace = () => (
    <>
      <div className="welcome-block">
        <div className="eyebrow">YOUR PERSONAL INTELLIGENCE</div>
        <h1>
          Think clearly.
          <br />
          <span>Decide consciously.</span>
        </h1>
        <p>
          An AI that learns your context, challenges your assumptions,
          and helps you make decisions with confidence.
        </p>
      </div>

      <div className="workspace-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker">AI WORKSPACE</span>
            <h2>What would you like to explore?</h2>
          </div>
          <span className="status-pill"><span /> Preview mode</span>
        </div>

        <div className="suggestion-grid">
          <button className="suggestion" onClick={() => setQuestion('Help me think through an important decision.')}>
            <span className="suggestion-icon">◇</span>
            <strong>Think through a decision</strong>
            <small>Explore options and trade-offs</small>
          </button>

          <button className="suggestion" onClick={() => setQuestion('Challenge my assumptions about this idea.')}>
            <span className="suggestion-icon">⌁</span>
            <strong>Challenge my thinking</strong>
            <small>Find assumptions and blind spots</small>
          </button>

          <button className="suggestion" onClick={() => setQuestion('Help me understand a complex topic.')}>
            <span className="suggestion-icon">✳</span>
            <strong>Understand something</strong>
            <small>Learn with personal context</small>
          </button>
        </div>

        <div className="composer">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Share what's on your mind..."
            rows={3}
          />
          <div className="composer-footer">
            <span>◉ Personal context · Private workspace</span>
            <button
              className="send-button"
              disabled={!question.trim()}
              onClick={() => setActiveSection('Transparency')}
              title="Backend connection will be added in the next stage"
            >
              Prepare question <span>→</span>
            </button>
          </div>
        </div>

        <p className="preview-note">
          This is a frontend preview. Questions are not sent to an AI model yet.
        </p>
      </div>

      <div className="lower-grid">
        <div className="info-card">
          <div className="card-icon purple">◎</div>
          <span className="section-kicker">TRANSPARENCY FIRST</span>
          <h3>Understand how AI responds</h3>
          <p>
            Explore the models, memories, data access and recorded events
            behind each interaction.
          </p>
          <button className="text-link" onClick={() => setActiveSection('Transparency')}>
            Explore transparency <span>→</span>
          </button>
        </div>

        <div className="info-card">
          <div className="card-icon blue">◫</div>
          <span className="section-kicker">YOUR MEMORY</span>
          <h3>You remain in control</h3>
          <p>
            Inspect, correct and manage the information your personal AI
            is allowed to remember.
          </p>
          <button className="text-link" onClick={() => setActiveSection('Memory')}>
            Explore memory <span>→</span>
          </button>
        </div>
      </div>
    </>
  )

  const sectionContent: Record<Exclude<Section, 'Workspace'>, {
    eyebrow: string
    title: string
    description: string
    items: string[]
  }> = {
    Transparency: {
      eyebrow: 'ACCOUNTABILITY & EXPLAINABILITY',
      title: 'Transparency Center',
      description: 'Understand what happened during an AI interaction, what information was used, and what the system can and cannot explain.',
      items: [
        'Interaction history and audit timeline',
        'Model and execution details',
        'Retrieved memories and privacy scope',
        'Limitations, uncertainty and recorded decisions',
      ],
    },
    Memory: {
      eyebrow: 'PERSONAL KNOWLEDGE',
      title: 'Memory Center',
      description: 'Your memory should belong to you. Review what the system knows and control how it is used.',
      items: [
        'Inspect stored personal memories',
        'Correct inaccurate information',
        'Approve information before sharing',
        'Manage and delete stored memories',
      ],
    },
    Privacy: {
      eyebrow: 'YOUR DATA. YOUR CONTROL.',
      title: 'Privacy & Security',
      description: 'Control the boundaries between personal information and shared knowledge.',
      items: [
        'Personal and public memory boundaries',
        'Data access and permission controls',
        'Local and external model execution',
        'Sharing and consent management',
      ],
    },
    Settings: {
      eyebrow: 'PREFERENCES',
      title: 'Account & Settings',
      description: 'Manage your profile and configure your CognitiveAI experience.',
      items: [
        'Account and profile',
        'Model preferences',
        'Interface preferences',
        'Security and account access',
      ],
    },
  }

  const renderSection = () => {
    if (activeSection === 'Workspace') return renderWorkspace()

    const page = sectionContent[activeSection]

    return (
      <div className="section-page">
        <div className="eyebrow">{page.eyebrow}</div>
        <h1>{page.title}</h1>
        <p className="section-description">{page.description}</p>

        <div className="feature-list">
          {page.items.map((item, index) => (
            <div className="feature-row" key={item}>
              <span className="feature-number">0{index + 1}</span>
              <span>{item}</span>
              <span className="feature-arrow">↗</span>
            </div>
          ))}
        </div>

        <div className="coming-soon">
          <span className="coming-icon">✳</span>
          <div>
            <strong>Being built with transparency in mind</strong>
            <p>
              This section is part of the interface foundation.
              Its live data and controls will be connected in upcoming
              development stages.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">C</div>
          <div>
            <strong>cognitive<span>ai</span></strong>
            <small>PERSONAL INTELLIGENCE</small>
          </div>
        </div>

        <div className="sidebar-label">PLATFORM</div>

        <nav className="navigation">
          {navigation.map((item) => (
            <button
              key={item.name}
              className={`nav-item ${activeSection === item.name ? 'active' : ''}`}
              onClick={() => setActiveSection(item.name)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.name}</span>
              {item.name === 'Transparency' && <span className="nav-dot" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="local-status">
            <span className="online-dot" />
            <div>
              <strong>Local environment</strong>
              <small>Development preview</small>
            </div>
          </div>

          <div className="profile">
            <div className="avatar">M</div>
            <div className="profile-text">
              <strong>Personal workspace</strong>
              <small>Local development</small>
            </div>
            <span className="profile-menu">···</span>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Platform</span>
            <span className="breadcrumb-separator">/</span>
            <strong>{activeSection}</strong>
          </div>

          <div className="topbar-right">
            <span className="environment-label">DEVELOPMENT</span>
            <button className="help-button" title="Help">?</button>
          </div>
        </header>

        <div className="page-content">
          {renderSection()}
        </div>

        <footer className="footer">
          <span>© 2026 CognitiveAI</span>
          <span>Built around trust, context and conscious decisions.</span>
        </footer>
      </main>
    </div>
  )
}

export default App

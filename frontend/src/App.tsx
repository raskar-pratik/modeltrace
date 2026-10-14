/**
 * Main ModelTrace Application Shell, Router, Keyboard Navigation, and Live Backend State Sync.
 */

import React, { useEffect, useState } from 'react';
import { apiClient, subscribeCache } from './api/client';
import { BranchingMIcon } from './components/BrandAndUI';
import { OverviewScreen } from './screens/OverviewScreen';
import { ProfilesScreen, ReferencesScreen } from './screens/ProfilesAndReferencesScreen';
import { NewAuditScreen, ProbesScreen } from './screens/ProbesAndNewAuditScreen';
import {
  AuditsScreen,
  EvidenceScreen,
  ReportsAndSettingsScreen,
} from './screens/AuditsAndEvidenceScreen';
import {
  AppSettings,
  Audit,
  AuditDetailBundle,
  Calibration,
  ProbeRevision,
  Profile,
  SystemCapabilities,
} from './types/api';

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', shortcut: '1' },
  { id: 'profiles', label: 'Endpoint Profiles', shortcut: '2' },
  { id: 'references', label: 'References & Calibration', shortcut: '3' },
  { id: 'probes', label: 'Probe Library', shortcut: '4' },
  { id: 'new-audit', label: 'New Audit', shortcut: '5' },
  { id: 'audits', label: 'Audit History', shortcut: '6' },
  { id: 'evidence', label: 'Evidence', shortcut: '7' },
  { id: 'reports', label: 'Reports & Settings', shortcut: '8' },
];

function getInitialScreenFromUrl(): string {
  const path = window.location.pathname.replace(/^\/+|\/+$/g, '');
  if (NAV_ITEMS.some((item) => item.id === path)) {
    return path;
  }
  return 'overview';
}

export function App() {
  const [activeScreen, setActiveScreen] = useState<string>(getInitialScreenFromUrl());
  const [selectedAuditId, setSelectedAuditId] = useState<string | null>(null);

  const [capabilities, setCapabilities] = useState<SystemCapabilities | null>(null);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [probeRevisions, setProbeRevisions] = useState<ProbeRevision[]>([]);
  const [calibrations, setCalibrations] = useState<Calibration[]>([]);
  const [audits, setAudits] = useState<Audit[]>([]);
  const [latestDetail, setLatestDetail] = useState<AuditDetailBundle | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [offlineError, setOfflineError] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState<string>('');

  const fetchAllWorkspaceData = async (bypassCache = false) => {
    try {
      setOfflineError(null);
      const [cap, setRes, profRes, revRes, calRes, audRes] = await Promise.all([
        apiClient.getCapabilities(bypassCache),
        apiClient.getSettings(bypassCache),
        apiClient.listProfiles(bypassCache),
        apiClient.listProbeRevisions(true, bypassCache),
        apiClient.listCalibrations(bypassCache),
        apiClient.listAudits(bypassCache),
      ]);
      setCapabilities(cap);
      setSettings(setRes);
      setProfiles(profRes.profiles || []);
      setProbeRevisions(revRes.revisions || []);
      setCalibrations(calRes.calibrations || []);
      const auditList = audRes.audits || [];
      setAudits(auditList);
      if (auditList.length > 0) {
        const targetId = selectedAuditId || auditList[0].id;
        if (!selectedAuditId) {
          setSelectedAuditId(targetId);
        }
        const det = await apiClient.getAuditDetail(auditList[0].id, bypassCache);
        setLatestDetail(det);
      } else {
        setLatestDetail(null);
      }
    } catch (err: any) {
      setOfflineError(err.detail || err.message || 'Backend connection error.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllWorkspaceData(false);
    const unsub = subscribeCache(() => {
      fetchAllWorkspaceData(false);
    });
    const onPopState = () => {
      setActiveScreen(getInitialScreenFromUrl());
    };
    window.addEventListener('popstate', onPopState);
    return () => {
      unsub();
      window.removeEventListener('popstate', onPopState);
    };
  }, []);

  const navigateTo = (screen: string, auditId?: string) => {
    if (auditId) {
      setSelectedAuditId(auditId);
    }
    setActiveScreen(screen);
    setAnnouncement(`Navigated to ${screen}`);
    const nextUrl = screen === 'overview' ? '/' : `/${screen}`;
    if (window.location.pathname !== nextUrl) {
      window.history.pushState({}, '', nextUrl);
    }
    window.scrollTo(0, 0);
  };

  return (
    <div className="app-shell">
      <div className="sr-only" aria-live="polite">
        {announcement}
      </div>

      <header className="top-header" role="banner">
        <div className="top-header-inner">
          <a
            className="brand-cluster"
            href="/"
            onClick={(e: any) => {
              e.preventDefault();
              navigateTo('overview');
            }}
          >
            <BranchingMIcon size={34} />
            <div>
              <div className="brand-wordmark">ModelTrace</div>
              <div className="brand-tagline">Know what your API is serving.</div>
            </div>
          </a>

          <nav className="nav-bar" aria-label="Primary Workbench Navigation">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`nav-tab ${activeScreen === item.id ? 'nav-tab-active' : ''}`}
                onClick={() => navigateTo(item.id)}
                data-testid={`nav-tab-${item.id}`}
              >
                <span>{item.label}</span>
                <kbd>{item.shortcut}</kbd>
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="main-container" role="main">
        {offlineError && (
          <div className="notice-banner notice-banner-error" role="alert" data-testid="offline-error-banner">
            <div>
              <strong>Backend Communication Issue:</strong> {offlineError}
            </div>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => fetchAllWorkspaceData(true)}
            >
              Retry Connection
            </button>
          </div>
        )}

        {loading && !capabilities ? (
          <div className="bento-card" style={{ padding: '48px', textAlign: 'center' }}>
            <div className="mono" style={{ color: '#787774' }}>
              Loading persisted ModelTrace workspace state...
            </div>
          </div>
        ) : (
          <div>
            {activeScreen === 'overview' && (
              <OverviewScreen
                audits={audits}
                latestDetail={latestDetail}
                capabilities={capabilities}
                settings={settings}
                onNavigate={navigateTo}
                onRefresh={() => fetchAllWorkspaceData(true)}
              />
            )}

            {activeScreen === 'profiles' && (
              <ProfilesScreen
                profiles={profiles}
                settings={settings}
                onRefresh={() => fetchAllWorkspaceData(true)}
              />
            )}

            {activeScreen === 'references' && (
              <ReferencesScreen
                profiles={profiles}
                calibrations={calibrations}
                probeRevisions={probeRevisions}
                capabilities={capabilities}
                settings={settings}
                onRefresh={() => fetchAllWorkspaceData(true)}
              />
            )}

            {activeScreen === 'probes' && (
              <ProbesScreen
                revisions={probeRevisions}
                onRefresh={() => fetchAllWorkspaceData(true)}
              />
            )}

            {activeScreen === 'new-audit' && (
              <NewAuditScreen
                profiles={profiles}
                probeRevisions={probeRevisions}
                calibrations={calibrations}
                settings={settings}
                onAuditCreated={(auditId) => {
                  fetchAllWorkspaceData(true);
                  navigateTo('audits', auditId);
                }}
              />
            )}

            {activeScreen === 'audits' && (
              <AuditsScreen
                audits={audits}
                selectedAuditId={selectedAuditId}
                settings={settings}
                onSelectAudit={(id) => setSelectedAuditId(id)}
                onNavigate={navigateTo}
                onRefresh={() => fetchAllWorkspaceData(true)}
              />
            )}

            {activeScreen === 'evidence' && (
              <EvidenceScreen
                audits={audits}
                selectedAuditId={selectedAuditId}
                settings={settings}
                onSelectAudit={(id) => setSelectedAuditId(id)}
              />
            )}

            {activeScreen === 'reports' && (
              <ReportsAndSettingsScreen
                audits={audits}
                selectedAuditId={selectedAuditId}
                settings={settings}
                capabilities={capabilities}
                onSelectAudit={(id) => setSelectedAuditId(id)}
                onRefresh={() => fetchAllWorkspaceData(true)}
              />
            )}
          </div>
        )}
      </main>
    </div>
  );
}

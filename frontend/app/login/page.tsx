"use client";

import { createClient } from "@/lib/supabase/client";
import { ShieldCheck, Quote, Lock, MessageCircleHeart } from "lucide-react";

export default function LoginPage() {
  const supabase = createClient();

  const handleGoogleLogin = async () => {
    await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${window.location.origin}/auth/callback`,
      },
    });
  };

  return (
    <main
      className="h-screen w-screen overflow-hidden relative bg-cover bg-center"
      style={{ backgroundImage: "url('/login-bg.png')" }}
    >
      <div className="relative z-10 h-full flex items-center justify-center p-4">
        <div className="w-full max-w-[1350px] max-h-[92vh] rounded-[24px] overflow-hidden">
          <div className="grid lg:grid-cols-2 h-full items-center gap-6">

            {/* ================= LEFT SIDE ================= */}
            <section className="relative px-6 lg:px-10 py-6 flex flex-col justify-center overflow-hidden">

              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#5937f5] to-[#7548ff] flex items-center justify-center shadow-lg shadow-purple-300/40">
                  <MessageCircleHeart className="text-white" size={22} strokeWidth={2} />
                </div>
                <div>
                  <h1 className="text-[24px] font-bold text-[#101c42] leading-none">MedRAG</h1>
                  <p className="text-[#617096] text-[13px] mt-0.5">Medical AI Assistant</p>
                </div>
              </div>

              <div className="max-w-[560px]">
                <h2 className="text-[34px] lg:text-[40px] leading-[1.1] font-bold tracking-[-1px] text-[#0c183b]">
                  Your trusted
                  <br />
                  medical companion
                  <span className="text-[#5c35f5]">.</span>
                </h2>

                <p className="mt-3 text-[16px] leading-6 text-[#405177] max-w-[480px]">
                  Get accurate, evidence-based answers from verified medical
                  sources with proper citations.
                </p>

                <Feature
                  icon={<ShieldCheck size={22} />}
                  title="Verified Sources"
                  description="Answers backed by NIH, MedlinePlus and PubMed data."
                />

                <Feature
                  icon={<Quote size={22} />}
                  title="Cited Answers"
                  description="Every response includes citations so you can trust the information."
                />

                <Feature
                  icon={<Lock size={22} />}
                  title="Secure & Private"
                  description="Your conversations are encrypted and never shared."
                />
              </div>
            </section>

            {/* ================= RIGHT SIDE ================= */}
            <section className="flex items-center justify-center px-4 lg:px-8">
              <div className="w-full max-w-[480px]">

                <div className="bg-white rounded-[20px] shadow-[0_25px_70px_rgba(50,70,130,0.12)] px-8 py-8">

                  <div className="flex justify-center mb-3">
                    <div className="w-12 h-12 rounded-full flex items-center justify-center">
                      <MessageCircleHeart size={40} strokeWidth={1.8} className="text-[#6540f5]" />
                    </div>
                  </div>

                  <div className="text-center">
                    <h2 className="text-[26px] font-bold text-[#101c42]">Welcome back</h2>
                    <p className="mt-1 text-[14px] text-[#657292]">Sign in to continue your conversations</p>
                  </div>

                  <div className="mt-6">
                    <button
                      type="button"
                      onClick={handleGoogleLogin}
                      className="w-full h-[50px] border border-[#d8deec] rounded-xl flex items-center justify-center gap-3 text-[15px] font-semibold text-[#101c42] hover:bg-[#f7f8fc] transition"
                    >
                      <GoogleIcon />
                      Continue with Google
                    </button>
                  </div>

                  <p className="text-center mt-5 text-xs text-[#8a96b1]">
                    By continuing, you agree to our Terms of Service and Privacy Policy.
                  </p>
                </div>

                <div className="flex items-center justify-center gap-6 mt-5 text-[#526284]">
                  <SecurityBadge icon={<Lock size={18} />} text="Privacy Protected" />
                  <SecurityBadge icon={<ShieldCheck size={18} />} text="Secure" />
                </div>
              </div>
            </section>

          </div>
        </div>
      </div>
    </main>
  );
}

/* ================= COMPONENTS ================= */

function Feature({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="flex items-start gap-3 mt-4">
      <div className="w-10 h-10 rounded-full bg-[#e9e5ff] flex items-center justify-center text-[#5b38ef] shrink-0">
        {icon}
      </div>
      <div>
        <h3 className="text-[14px] font-bold text-[#111d40]">{title}</h3>
        <p className="mt-0.5 text-[13px] leading-5 text-[#506080] max-w-[380px]">{description}</p>
      </div>
    </div>
  );
}

function SecurityBadge({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[#5c3af1]">{icon}</span>
      <span className="text-xs font-medium">{text}</span>
    </div>
  );
}

function GoogleIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24">
      <path fill="#4285F4" d="M21.35 12.23c0-.79-.07-1.55-.22-2.27H12v4.3h5.23a4.47 4.47 0 0 1-1.94 2.93v2.42h3.14c1.84-1.69 2.92-4.18 2.92-7.38Z" />
      <path fill="#34A853" d="M12 21.5c2.63 0 4.84-.87 6.45-2.36l-3.14-2.42c-.87.58-1.98.92-3.31.92-2.54 0-4.69-1.72-5.46-4.03H3.29v2.5A9.74 9.74 0 0 0 12 21.5Z" />
      <path fill="#FBBC05" d="M6.54 13.61A5.86 5.86 0 0 1 6.23 12c0-.56.1-1.1.31-1.61V7.89H3.29A9.73 9.73 0 0 0 2.25 12c0 1.57.38 3.05 1.04 4.11l3.25-2.5Z" />
      <path fill="#EA4335" d="M12 6.36c1.43 0 2.7.49 3.71 1.45l2.78-2.78C16.84 3.46 14.63 2.5 12 2.5a9.74 9.74 0 0 0-8.71 5.39l3.25 2.5C7.31 8.08 9.46 6.36 12 6.36Z" />
    </svg>
  );
}
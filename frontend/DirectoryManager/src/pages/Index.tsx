import Header from "@/components/Header";
import SecurityCard from "@/components/SecurityCard";
import surveillanceHero from "@/assets/surveillance-hero.jpg";
import entryExitHero from "@/assets/entry-exit-hero.jpg";
import techBackground from "@/assets/tech-background.jpg";

const Index = () => {
  return (
    <div 
      className="min-h-screen tech-background relative"
      style={{
        backgroundImage: `linear-gradient(135deg, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0.6) 100%), url(${techBackground})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed',
      }}
    >
      {/* Header */}
      <Header />
      
      {/* Main Content */}
      <main className="container mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 leading-tight">
            Secure<span className="text-primary-glow">Vista</span>
          </h1>
          <p className="text-xl text-secondary max-w-2xl mx-auto leading-relaxed">
            Advanced security solutions powered by intelligent technology. 
            Monitor, control, and protect with cutting-edge surveillance systems.
          </p>
        </div>

        {/* Security Cards */}
        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          <SecurityCard
            title="Surveillance System"
            backgroundImage={surveillanceHero}
            onClick={() => window.location.href = 'http://localhost:5000'}
          />
          <SecurityCard
            title="Entry Exit System"
            backgroundImage={entryExitHero}
            onClick={() => window.location.href = 'http://localhost:8080'}
          />
        </div>

        {/* Tech Features */}
        <div className="mt-20 text-center">
          <div className="grid md:grid-cols-3 gap-8 max-w-3xl mx-auto">
            <div className="p-6 rounded-lg bg-card/20 backdrop-blur-sm border border-border/30">
              <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center mx-auto mb-4">
                <div className="w-6 h-6 bg-primary-glow rounded-full"></div>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">24/7 Monitoring</h3>
              <p className="text-secondary text-sm">Continuous surveillance with real-time alerts</p>
            </div>
            
            <div className="p-6 rounded-lg bg-card/20 backdrop-blur-sm border border-border/30">
              <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center mx-auto mb-4">
                <div className="w-6 h-6 bg-primary-glow rounded-full"></div>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">AI Detection</h3>
              <p className="text-secondary text-sm">Smart recognition and threat analysis</p>
            </div>
            
            <div className="p-6 rounded-lg bg-card/20 backdrop-blur-sm border border-border/30">
              <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center mx-auto mb-4">
                <div className="w-6 h-6 bg-primary-glow rounded-full"></div>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Secure Access</h3>
              <p className="text-secondary text-sm">Advanced entry control systems</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;

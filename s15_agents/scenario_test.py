"""
S15 — Scenario Tests
Tests de scénarios réalistes pour l'agent
"""

import sys
from agent import ReactAgent
from tools import get_tools
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ScenarioTester:
    """Testeur de scénarios pour l'agent"""
    
    def __init__(self):
        self.agent = ReactAgent(verbose=False)
        self.results = []
    
    def run_scenario(self, name: str, task: str, expected_keywords: list):
        """
        Exécuter un scénario de test
        
        Args:
            name: Nom du scénario
            task: Tâche à accomplir
            expected_keywords: Mots-clés attendus dans la réponse
        """
        print(f"\n{'='*60}")
        print(f"🧪 SCÉNARIO: {name}")
        print(f"{'='*60}")
        print(f"Tâche: {task}\n")
        
        try:
            # Exécuter l'agent
            result = self.agent.run(task)
            
            print(f"✨ Résultat:\n{result}\n")
            
            # Vérifier les mots-clés
            result_lower = result.lower()
            found_keywords = [kw for kw in expected_keywords if kw.lower() in result_lower]
            
            success = len(found_keywords) > 0
            
            if success:
                print(f"✅ SUCCÈS - Mots-clés trouvés: {found_keywords}")
            else:
                print(f"⚠️  ATTENTION - Aucun mot-clé trouvé parmi: {expected_keywords}")
            
            self.results.append({
                "name": name,
                "success": success,
                "result": result
            })
            
            return success
            
        except Exception as e:
            print(f"❌ ERREUR: {e}")
            self.results.append({
                "name": name,
                "success": False,
                "error": str(e)
            })
            return False
    
    def print_summary(self):
        """Afficher le résumé des tests"""
        print(f"\n{'='*60}")
        print(f"📊 RÉSUMÉ DES TESTS")
        print(f"{'='*60}")
        
        total = len(self.results)
        successes = sum(1 for r in self.results if r["success"])
        
        print(f"\nTotal: {total} scénarios")
        print(f"Succès: {successes}")
        print(f"Échecs: {total - successes}")
        print(f"Taux de réussite: {successes/total*100:.1f}%")
        
        print(f"\nDétails:")
        for i, result in enumerate(self.results, 1):
            status = "✅" if result["success"] else "❌"
            print(f"{i}. {status} {result['name']}")


# ============================================================
# SCÉNARIOS DE TEST
# ============================================================

def test_scenario_1_travel_planning():
    """Scénario 1: Planification de voyage avec budget"""
    tester = ScenarioTester()
    
    tester.run_scenario(
        name="Travel Planning - Budget",
        task="Planifie un voyage à Paris avec un budget de 1000€. Estime les coûts de transport et hébergement.",
        expected_keywords=["paris", "€", "budget", "transport", "hôtel", "coût"]
    )
    
    return tester


def test_scenario_2_complex_calculation():
    """Scénario 2: Calcul complexe avec remise"""
    tester = ScenarioTester()
    
    tester.run_scenario(
        name="Complex Calculation - Discount",
        task="Un produit coûte 49€. Il y a une réduction de 20%. Quel est le prix final?",
        expected_keywords=["39", "euro", "€", "prix"]
    )
    
    return tester


def test_scenario_3_multi_step_research():
    """Scénario 3: Recherche multi-étapes"""
    tester = ScenarioTester()
    
    tester.run_scenario(
        name="Multi-step Research",
        task="Quelle est la capitale de la France et combien d'habitants y vivent?",
        expected_keywords=["paris", "capitale", "habitants", "million"]
    )
    
    return tester


def test_scenario_4_currency_conversion():
    """Scénario 4: Conversion de devises"""
    tester = ScenarioTester()
    
    tester.run_scenario(
        name="Currency Conversion",
        task="Convertis 500 EUR en USD. Quel est le montant?",
        expected_keywords=["usd", "dollar", "550", "montant"]
    )
    
    return tester


def test_scenario_5_time_query():
    """Scénario 5: Requête temporelle"""
    tester = ScenarioTester()
    
    tester.run_scenario(
        name="Time Query",
        task="Quelle est l'heure et la date actuelles?",
        expected_keywords=["202", ":", "heure", "date"]
    )
    
    return tester


def test_scenario_6_price_comparison():
    """Scénario 6: Comparaison de prix"""
    tester = ScenarioTester()
    
    tester.run_scenario(
        name="Price Comparison",
        task="Si un billet Paris-Londres coûte 120€ et un hôtel 80€ par nuit pour 2 nuits, quel est le coût total?",
        expected_keywords=["280", "€", "total", "coût"]
    )
    
    return tester


def test_all_scenarios():
    """Exécuter tous les scénarios"""
    print("\n" + "="*60)
    print("🧪 TEST DE TOUS LES SCÉNARIOS")
    print("="*60)
    
    # Créer un tester global
    global_tester = ScenarioTester()
    
    # Scénarios
    scenarios = [
        ("Travel Planning", 
         "Planifie un voyage à Paris avec un budget de 1000€. Estime les coûts.",
         ["paris", "€", "budget", "coût"]),
        
        ("Discount Calculation",
         "Un produit coûte 49€ avec -20%. Prix final?",
         ["39", "€", "prix"]),
        
        ("Capital & Population",
         "Capitale de France et sa population?",
         ["paris", "million", "habitants"]),
        
        ("Currency Conversion",
         "Convertis 500 EUR en USD",
         ["usd", "dollar"]),
        
        ("Current Time",
         "Quelle heure est-il?",
         ["202", "heure"]),
        
        ("Budget Calculation",
         "Billet 120€ + hôtel 80€ x 2 nuits = total?",
         ["280", "€", "total"])
    ]
    
    for name, task, keywords in scenarios:
        global_tester.run_scenario(name, task, keywords)
        print("\n" + "-"*60)
    
    # Afficher le résumé
    global_tester.print_summary()
    
    return global_tester


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def main():
    """Fonction principale"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Tests de scénarios pour l'agent")
    parser.add_argument('--all', action='store_true', help='Exécuter tous les scénarios')
    parser.add_argument('--scenario', type=int, help='Numéro du scénario à exécuter (1-6)')
    parser.add_argument('--verbose', action='store_true', help='Mode verbose')
    parser.add_argument('--benchmark', action='store_true', help='Mode benchmark (tous les tests)')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
    
    if args.all or args.benchmark:
        test_all_scenarios()
    elif args.scenario:
        scenarios = [
            test_scenario_1_travel_planning,
            test_scenario_2_complex_calculation,
            test_scenario_3_multi_step_research,
            test_scenario_4_currency_conversion,
            test_scenario_5_time_query,
            test_scenario_6_price_comparison
        ]
        
        if 1 <= args.scenario <= len(scenarios):
            tester = scenarios[args.scenario - 1]()
            tester.print_summary()
        else:
            print(f"❌ Scénario invalide. Choisissez entre 1 et {len(scenarios)}")
    else:
        # Menu interactif
        print("\n" + "="*60)
        print("🧪 TESTS DE SCÉNARIOS")
        print("="*60)
        print("\nScénarios disponibles:")
        print("1. Travel Planning (voyage avec budget)")
        print("2. Complex Calculation (calcul avec remise)")
        print("3. Multi-step Research (capitale + population)")
        print("4. Currency Conversion (conversion devise)")
        print("5. Time Query (heure actuelle)")
        print("6. Price Comparison (comparaison prix)")
        print("7. Exécuter TOUS les scénarios")
        print("0. Quitter")
        
        choice = input("\nVotre choix: ").strip()
        
        if choice == "0":
            print("👋 Au revoir!")
            return
        elif choice == "7":
            test_all_scenarios()
        elif choice in ["1", "2", "3", "4", "5", "6"]:
            scenarios = [
                test_scenario_1_travel_planning,
                test_scenario_2_complex_calculation,
                test_scenario_3_multi_step_research,
                test_scenario_4_currency_conversion,
                test_scenario_5_time_query,
                test_scenario_6_price_comparison
            ]
            tester = scenarios[int(choice) - 1]()
            tester.print_summary()
        else:
            print("❌ Choix invalide")


if __name__ == "__main__":
    main()

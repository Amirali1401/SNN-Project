import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# ۱. ساختار ذره و سنجش غلبه پارتو
# ==========================================
class Particle:
    def __init__(self, num_features):
        self.position = np.random.rand(num_features)
        self.velocity = np.zeros(num_features)
        self.best_position = np.copy(self.position)
        self.best_objectives = [float('inf'), float('inf')]
        self.objectives = [float('inf'), float('inf')]

    def get_binary_mask(self):
        return self.position > 0.5

def dominates(obj1, obj2):
    not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
    strictly_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    return not_worse and strictly_better

# ==========================================
# ۲. کلاس الگوریتم MOPSO برای انتخاب ژن
# ==========================================
class MOPSOFeatureSelection:
    def __init__(self, num_particles=30, max_iter=20, archive_size=30):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.archive_size = archive_size
        self.w, self.c1, self.c2 = 0.5, 1.5, 1.5

    def evaluate(self, particle, X, y):
        mask = particle.get_binary_mask()
        num_selected = np.sum(mask)
        
        if num_selected == 0:
            particle.objectives = [1.0, 1.0]
            return

        clf = RandomForestClassifier(n_estimators=30, random_state=42)
        scores = cross_val_score(clf, X[:, mask], y, cv=3, scoring='accuracy')
        
        error_rate = 1.0 - np.mean(scores)
        feature_ratio = num_selected / X.shape[1]
        particle.objectives = [error_rate, feature_ratio]

    def update_archive(self, swarm, archive):
        combined = archive + swarm
        non_dominated = []
        for p in combined:
            if not any(dominates(other.objectives, p.objectives) for other in combined if p != other):
                if not any(np.array_equal(p.objectives, nd.objectives) for nd in non_dominated):
                    non_dominated.append(p)
        return non_dominated[:self.archive_size]

    def run(self, X, y):
        num_features = X.shape[1]
        swarm = [Particle(num_features) for _ in range(self.num_particles)]
        archive = []

        for p in swarm:
            self.evaluate(p, X, y)
            p.best_position = np.copy(p.position)
            p.best_objectives = list(p.objectives)

        archive = self.update_archive(swarm, archive)

        for it in range(1, self.max_iter + 1):
            leader = np.random.choice(archive) if len(archive) > 0 else np.random.choice(swarm)
            
            for p in swarm:
                r1, r2 = np.random.rand(num_features), np.random.rand(num_features)
                p.velocity = (self.w * p.velocity + 
                              self.c1 * r1 * (p.best_position - p.position) + 
                              self.c2 * r2 * (leader.position - p.position))
                p.position = np.clip(p.position + p.velocity, 0.0, 1.0)
                
                self.evaluate(p, X, y)
                if dominates(p.objectives, p.best_objectives):
                    p.best_position = np.copy(p.position)
                    p.best_objectives = list(p.objectives)

            archive = self.update_archive(swarm, archive)
            print(f"تکرار {it:02d}/{self.max_iter:02d} | تعداد پاسخ‌های پارتو: {len(archive)}")

        return archive

def extract_influential_genes(csv_path):
    df = pd.read_csv(csv_path)

    # اگر ژن‌ها در سطرها قرار دارند، جدول ترانهاده (ماتریس معکوس) می‌شود
    if df.shape[0] > 1000:
        df = df.set_index(df.columns[0]).T

    # جداسازی ویژگی‌های عددی بیان ژن و برچسب بیماران
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    gene_names = df.columns[:-1].tolist()

    print("="*60)
    print(f"دیتاست بارگذاری شد | تعداد نمونه بیماران: {X.shape[0]} | تعداد کل ژن‌ها: {X.shape[1]}")
    print("="*60)

    # اجرای الگوریتم MOPSO
    mopso = MOPSOFeatureSelection(num_particles=25, max_iter=20)
    pareto_front = mopso.run(X, y)

    best_solution = min(pareto_front, key=lambda p: p.objectives[0])
    selected_mask = best_solution.get_binary_mask()
    selected_genes = np.array(gene_names)[selected_mask].tolist()

    print("\n" + "="*60)
    print("نتایج نهایی استخراج ژن‌های تاثیرگذار:")
    print("="*60)
    print(f"تعداد ژن‌های انتخاب‌شده: {len(selected_genes)} از {len(gene_names)}")
    print("-" * 60)
    for idx, gene in enumerate(selected_genes, 1):
        print(f"{idx:02d}. {gene}")
    print("="*60)

    return selected_genes


if __name__ == "__main__":
    # آدرس فایل CSV خود را اینجا قرار دهید
    csv_file_name = "GPL570-55999.csv"
    
    selected_genes_list = extract_influential_genes(csv_file_name)
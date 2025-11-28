"""
Standalone script to generate Bag of Words diagram
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def draw_bag_of_words_diagram(save_path='bag_of_words_diagram.png'):
    """
    Create a visual diagram explaining Bag of Words representation.
    """
    # Sample documents
    documents = [
        "I love this movie",
        "This movie is great",
        "I hate this movie"
    ]
    
    # Create vocabulary from all documents
    all_words = []
    for doc in documents:
        words = doc.lower().split()
        all_words.extend(words)
    
    vocabulary = sorted(list(set(all_words)))
    
    # Create BoW vectors
    bow_vectors = []
    for doc in documents:
        words = doc.lower().split()
        word_counts = Counter(words)
        vector = [word_counts.get(word, 0) for word in vocabulary]
        bow_vectors.append(vector)
    
    # Create the figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Documents section (left side)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.axis('off')
    ax1.text(0.5, 0.95, 'Documents', ha='center', va='top', fontsize=16, fontweight='bold', transform=ax1.transAxes)
    
    y_pos = 0.85
    for i, doc in enumerate(documents):
        ax1.text(0.1, y_pos, f'Doc {i+1}:', ha='left', va='top', fontsize=12, fontweight='bold', transform=ax1.transAxes)
        ax1.text(0.1, y_pos - 0.08, f'"{doc}"', ha='left', va='top', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), transform=ax1.transAxes)
        y_pos -= 0.2
    
    # 2. Vocabulary section (middle top)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.text(0.5, 0.9, 'Vocabulary', ha='center', va='top', fontsize=16, fontweight='bold', transform=ax2.transAxes)
    
    vocab_text = ' | '.join([f'{i}: {word}' for i, word in enumerate(vocabulary)])
    ax2.text(0.5, 0.5, vocab_text, ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5), transform=ax2.transAxes)
    
    # 3. Bag of Words Matrix (right side)
    ax3 = fig.add_subplot(gs[0:2, 1:3])
    
    # Create matrix visualization
    matrix = np.array(bow_vectors)
    im = ax3.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(max(row) for row in bow_vectors))
    
    # Set ticks and labels
    ax3.set_xticks(range(len(vocabulary)))
    ax3.set_xticklabels(vocabulary, rotation=45, ha='right')
    ax3.set_yticks(range(len(documents)))
    ax3.set_yticklabels([f'Doc {i+1}' for i in range(len(documents))])
    ax3.set_xlabel('Vocabulary Terms', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Documents', fontsize=12, fontweight='bold')
    ax3.set_title('Bag of Words Feature Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Add text annotations for each cell
    for i in range(len(documents)):
        for j in range(len(vocabulary)):
            text = ax3.text(j, i, matrix[i, j], ha="center", va="center", 
                          color="black", fontweight='bold', fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Word Count', rotation=270, labelpad=15)
    
    # 4. Example vectors (bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    ax4.text(0.5, 0.9, 'Feature Vectors (Bag of Words Representation)', ha='center', va='top', 
            fontsize=14, fontweight='bold', transform=ax4.transAxes)
    
    y_start = 0.7
    for i, (doc, vector) in enumerate(zip(documents, bow_vectors)):
        vector_str = '[' + ', '.join(map(str, vector)) + ']'
        ax4.text(0.05, y_start, f'Doc {i+1}:', ha='left', va='top', fontsize=11, fontweight='bold', 
                transform=ax4.transAxes)
        ax4.text(0.15, y_start, vector_str, ha='left', va='top', fontsize=10, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), 
                transform=ax4.transAxes)
        ax4.text(0.6, y_start, f'"{doc}"', ha='left', va='top', fontsize=10, 
                style='italic', transform=ax4.transAxes)
        y_start -= 0.15
    
    # Add explanation text
    explanation = (
        "Bag of Words: Each document is represented as a vector where each dimension "
        "corresponds to a word in the vocabulary. The value indicates how many times "
        "that word appears in the document."
    )
    ax4.text(0.5, 0.05, explanation, ha='center', va='bottom', fontsize=10, 
            style='italic', wrap=True, transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle('Bag of Words (BoW) Representation', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Bag of Words diagram saved to {save_path}")

if __name__ == "__main__":
    draw_bag_of_words_diagram()


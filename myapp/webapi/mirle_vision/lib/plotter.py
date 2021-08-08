from typing import List, Dict, Tuple, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Plotter:

    @staticmethod
    def plot_loss_curve(global_batches: List[int],
                        legend_to_losses_and_color_dict: Dict[str, Tuple[List[float], str]],
                        path_to_plot: str):
        global_batches_length = len(global_batches)
        for losses, _ in legend_to_losses_and_color_dict.values():
            assert global_batches_length == len(losses)

        matplotlib.use('agg')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10.24, 7.68))

        for losses, color in legend_to_losses_and_color_dict.values():
            ax.plot(global_batches, losses, color)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        legends = list(legend_to_losses_and_color_dict.keys())
        ax.legend(legends, loc='upper left', bbox_to_anchor=(1, 1), shadow=True)

        ax.set_xlabel('Global Batch')
        ax.set_ylabel('Loss')
        ax.set_ylim(bottom=0)
        ax.set_title('Loss Curve')
        ax.grid()

        fig.savefig(path_to_plot)
        plt.close()

    @staticmethod
    def plot_roc_curve(num_classes: int,
                       class_to_category_dict: Dict[int, str],
                       macro_average_auc: float,
                       class_to_auc_dict: Dict[int, float],
                       class_to_fpr_array_dict: Dict[int, np.ndarray],
                       class_to_tpr_array_dict: Dict[int, np.ndarray],
                       path_to_plot: str):
        matplotlib.use('agg')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10.24, 7.68))

        for c in range(1, num_classes):
            fpr_array = class_to_fpr_array_dict[c]
            tpr_array = class_to_tpr_array_dict[c]
            ax.plot(fpr_array, tpr_array)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        category_list = [class_to_category_dict[c] for c in range(1, num_classes)]
        auc_list = [class_to_auc_dict[c] for c in range(1, num_classes)]
        legends = [f'{c}: {a:.4f}' for c, a in zip(category_list, auc_list)]
        ax.legend(legends, loc='upper left', bbox_to_anchor=(1, 1), shadow=True)

        plt.xticks(np.arange(0, 1.01, .1))
        plt.yticks(np.arange(0, 1.01, .1))
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title(f'Receiver Operating Characteristic Curve:\nmacro-average AUC = {macro_average_auc:.4f}')
        ax.grid()

        fig.savefig(path_to_plot)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray,
                              class_to_category_dict: Dict[int, str],
                              path_to_plot: str):
        matplotlib.use('agg')
        categories = [v for k, v in class_to_category_dict.items() if k > 0]
        ax = sns.heatmap(data=confusion_matrix, cmap='YlGnBu', annot=True, fmt='d',
                         xticklabels=categories, yticklabels=categories)
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')
        fig = ax.get_figure()
        fig.savefig(path_to_plot)
        plt.close()

    @staticmethod
    def plot_pr_curve(num_classes: int,
                      class_to_category_dict: Dict[int, str],
                      mean_ap: float,
                      class_to_ap_dict: Dict[int, float],
                      class_to_inter_recall_array_dict: Dict[int, np.ndarray],
                      class_to_inter_precision_array_dict: Dict[int, np.ndarray],
                      path_to_plot: str):
        matplotlib.use('agg')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10.24, 7.68))

        for c in range(1, num_classes):
            inter_recall_array = class_to_inter_recall_array_dict[c]
            inter_precision_array = class_to_inter_precision_array_dict[c]
            ax.plot(inter_recall_array, inter_precision_array)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        category_list = [class_to_category_dict[c] for c in range(1, num_classes)]
        ap_list = [class_to_ap_dict[c] for c in range(1, num_classes)]
        legends = [f'{c}: {a:.4f}' for c, a in zip(category_list, ap_list)]
        ax.legend(legends, loc='upper left', bbox_to_anchor=(1, 1), shadow=True)

        plt.xticks(np.arange(0, 1.01, .1))
        plt.yticks(np.arange(0, 1.01, .1))
        ax.set_xlabel('Recall')
        ax.set_ylabel('Interpolated Precision')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title(f'Precision-Recall Curve: mean AP = {mean_ap:.4f}')
        ax.grid()

        fig.savefig(path_to_plot)
        plt.close()

    @staticmethod
    def plot_thresh_vs_pr_bar(num_classes: int,
                              class_to_category_dict: Dict[int, str],
                              class_to_ap_dict: Dict[int, float],
                              class_to_recall_array_dict: Dict[int, np.ndarray],
                              class_to_precision_array_dict: Dict[int, np.ndarray],
                              class_to_f1_score_array_dict: Dict[int, np.ndarray],
                              class_to_prob_array_dict: Dict[int, np.ndarray],
                              path_to_placeholder_to_plot: str):

        def plot(plotting_recall_array_: np.ndarray, plotting_precision_array_: np.ndarray,
                 plotting_f1_score_array_: np.ndarray, plotting_prob_array_: np.ndarray,
                 category: str, ap: float, path_to_plot: str):
            matplotlib.use('agg')
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12.80, 7.68))

            bar_width = 0.2
            pos = np.arange(plotting_prob_array_.shape[0])

            ax.bar(pos - bar_width, plotting_precision_array_, bar_width, color='blue', edgecolor='none')
            ax.bar(pos, plotting_recall_array_, bar_width, color='green', edgecolor='none')
            ax.bar(pos + bar_width, plotting_f1_score_array_, bar_width, color='purple', edgecolor='none')

            for i, p in enumerate(pos):
                ax.text(p - bar_width, plotting_precision_array_[i] + .002, f'{plotting_precision_array_[i]:.3f}', color='blue',
                        fontsize=6, rotation=90, ha='center', va='bottom')
                ax.text(p + .05, plotting_recall_array_[i] + .002, f'{plotting_recall_array_[i]:.3f}', color='green',
                        fontsize=6, rotation=90, ha='center', va='bottom')
                ax.text(p + bar_width + .05, plotting_f1_score_array_[i] + .002, f'{plotting_f1_score_array_[i]:.3f}', color='purple',
                        fontsize=6, rotation=90, ha='center', va='bottom')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

            legends = ['Precision', 'Recall', 'F1-Score']
            ax.legend(legends, loc='upper left', bbox_to_anchor=(1, 1), shadow=True)

            ax.set_xlabel('Confidence Threshold')
            plt.xticks(pos, [f'{it:.4f}' for it in plotting_prob_array_], rotation=45)
            ax.set_ylim([0.0, 1.1])  # leave space on the top
            ax.set_title(f'Threshold versus PR: {category} AP = {ap:.4f}')
            ax.grid()

            fig.savefig(path_to_plot)
            plt.close()

        num_quantities = 20
        quantized_thresh_array = np.floor(np.linspace(1, 0, num_quantities + 1) * num_quantities) / num_quantities

        class_to_quantized_recall_array_dict = {}
        class_to_quantized_precision_array_dict = {}
        class_to_quantized_f1_score_array_dict = {}

        for c in range(1, num_classes):
            recall_array = class_to_recall_array_dict[c]
            precision_array = class_to_precision_array_dict[c]
            f1_score_array = class_to_f1_score_array_dict[c]
            prob_array = class_to_prob_array_dict[c]

            # NOTE: Quantize into `num_quantities` bins from 0 to 1,
            #           for example:
            #               num_quantities = 20
            #               prob_array = 0.732  0.675  0.653  0.621  0.531  0.519
            #     quantized_prob_array = 0.70   0.65   0.65   0.60   0.50   0.50
            quantized_prob_array = np.floor(prob_array * num_quantities) / num_quantities

            # NOTE: Example as below
            #
            #                  prob_array =        0.70   0.65   0.65   0.60   0.50   0.50
            #                recall_array =        0.124  0.336  0.381  0.433  0.587  0.590
            #             precision_array =        0.883  0.707  0.733  0.684  0.512  0.506
            #              f1_score_array =        0.217  0.456  0.501  0.530  0.547  0.545
            #
            #    np.append(0, prob_array) = 0.00   0.70   0.65   0.65   0.60   0.50   0.50
            #    np.append(prob_array, 0) = 0.70   0.65   0.65   0.60   0.50   0.50   0.00
            #             unequal compare =    T      T      F      F      T      F      T
            #                        mask =           T      F      F      T      F      T
            #
            #            prob_array[mask] =        0.70                 0.60          0.50
            #          recall_array[mask] =        0.124                0.433         0.590
            #       precision_array[mask] =        0.883                0.684         0.506
            #        f1_score_array[mask] =        0.217                0.530         0.545
            #
            #    result keep only if [n]-th element is not equal to [n+1]-th element
            mask = (np.append(0, quantized_prob_array) != np.append(quantized_prob_array, 0))[1:]
            masked_recall_array = recall_array[mask]
            masked_precision_array = precision_array[mask]
            masked_f1_score_array = f1_score_array[mask]
            masked_prob_array = quantized_prob_array[mask]

            masked_recall_array = np.insert(masked_recall_array, 0, 0.)
            masked_precision_array = np.insert(masked_precision_array, 0, 0.)
            masked_f1_score_array = np.insert(masked_f1_score_array, 0, 0.)
            masked_prob_array = np.insert(masked_prob_array, 0, 1.01)

            quantized_recall_array = []
            quantized_precision_array = []
            quantized_f1_score_array = []
            quantized_prob_array = []

            for thresh in quantized_thresh_array:
                idx = (masked_prob_array >= thresh).nonzero()[0][-1]
                quantized_recall_array.append(masked_recall_array[idx])
                quantized_precision_array.append(masked_precision_array[idx])
                quantized_f1_score_array.append(masked_f1_score_array[idx])
                quantized_prob_array.append(thresh)

            quantized_recall_array = np.array(quantized_recall_array)
            quantized_precision_array = np.array(quantized_precision_array)
            quantized_f1_score_array = np.array(quantized_f1_score_array)
            quantized_prob_array = np.array(quantized_prob_array)

            class_to_quantized_recall_array_dict[c] = quantized_recall_array
            class_to_quantized_precision_array_dict[c] = quantized_precision_array
            class_to_quantized_f1_score_array_dict[c] = quantized_f1_score_array

            plotting_recall_array = quantized_recall_array
            plotting_precision_array = quantized_precision_array
            plotting_f1_score_array = quantized_f1_score_array
            plotting_prob_array = quantized_prob_array

            # NOTE: The metric point at top f1-score is useful information when choosing a threshold for the specific class;
            #       however, it can be lost after quantization (in the above example, the best f1-score 0.547 was gone),
            #       hence we are going to put it back here
            if f1_score_array.shape[0] > 0:
                top_f1_score_index = f1_score_array.argmax().item()
                recall_at_top_f1_score = recall_array[top_f1_score_index]
                precision_at_top_f1_score = precision_array[top_f1_score_index]
                f1_score_at_top_f1_score = f1_score_array[top_f1_score_index]
                prob_at_top_f1_score = prob_array[top_f1_score_index]

                inserting_index = np.digitize(prob_at_top_f1_score, quantized_prob_array)
                plotting_recall_array = np.insert(plotting_recall_array, inserting_index, recall_at_top_f1_score)
                plotting_precision_array = np.insert(plotting_precision_array, inserting_index, precision_at_top_f1_score)
                plotting_f1_score_array = np.insert(plotting_f1_score_array, inserting_index, f1_score_at_top_f1_score)
                plotting_prob_array = np.insert(plotting_prob_array, inserting_index, prob_at_top_f1_score)

            plot(plotting_recall_array_=plotting_recall_array,
                 plotting_precision_array_=plotting_precision_array,
                 plotting_f1_score_array_=plotting_f1_score_array,
                 plotting_prob_array_=plotting_prob_array,
                 category=class_to_category_dict[c],
                 ap=class_to_ap_dict[c],
                 path_to_plot=path_to_placeholder_to_plot.format(c))

        plot(
            plotting_recall_array_=np.stack([class_to_quantized_recall_array_dict[c] for c in range(1, num_classes)], axis=0).mean(axis=0),
            plotting_precision_array_=np.stack([class_to_quantized_precision_array_dict[c] for c in range(1, num_classes)], axis=0).mean(axis=0),
            plotting_f1_score_array_=np.stack([class_to_quantized_f1_score_array_dict[c] for c in range(1, num_classes)], axis=0).mean(axis=0),
            plotting_prob_array_=quantized_thresh_array,
            category='mean',
            ap=np.mean([class_to_ap_dict[c] for c in range(1, num_classes)], axis=0).item(),
            path_to_plot=path_to_placeholder_to_plot.format('mean')
        )

    @staticmethod
    def plot_2d_scatter_with_histogram(labels: List[str],
                                       label_to_x_data_dict: Dict[str, List],
                                       label_to_y_data_dict: Dict[str, List],
                                       title: str,
                                       on_pick_callback: Callable = None,
                                       label_to_pick_info_data_dict: Dict[str, List] = None):
        num_labels = len(labels)
        is_pickable = on_pick_callback is not None

        assert len(label_to_x_data_dict) == num_labels
        assert len(label_to_y_data_dict) == num_labels
        if is_pickable:
            assert label_to_pick_info_data_dict is not None and len(label_to_pick_info_data_dict) == num_labels

        all_x_data = [x for x_data in label_to_x_data_dict.values() for x in x_data]
        all_y_data = [y for y_data in label_to_y_data_dict.values() for y in y_data]
        grid = sns.jointplot(x=all_x_data, y=all_y_data, kind='reg', scatter=False)

        scatter_to_pick_info_data = {}
        for label in labels:
            x_data = label_to_x_data_dict[label]
            y_data = label_to_y_data_dict[label]
            scatter = grid.ax_joint.scatter(x=x_data, y=y_data, label=label, picker=is_pickable)
            if is_pickable:
                pick_info_data = label_to_pick_info_data_dict[label]
                scatter_to_pick_info_data[scatter] = pick_info_data

        grid.set_axis_labels(xlabel='Width', ylabel='Height')
        grid.ax_joint.set_title(title)
        grid.ax_joint.legend()
        fig = grid.fig
        fig.tight_layout()

        if is_pickable:
            def on_pick(event):
                index = event.ind[0]
                pick_info = scatter_to_pick_info_data[event.artist][index]
                on_pick_callback(pick_info)
            fig.canvas.mpl_connect('pick_event', on_pick)

        plt.show()
        plt.close()

    @staticmethod
    def plot_category_vs_count_bar(category_vs_count_dict: Dict[str, int]):
        categories = [k for k in category_vs_count_dict.keys()]
        counts = [v for v in category_vs_count_dict.values()]
        category_and_count_list = [(category, count)
                                   for count, category in sorted(zip(counts, categories), reverse=True)]

        ax = sns.barplot(x=[category for category, _ in category_and_count_list],
                         y=[count for _, count in category_and_count_list])
        for patch in ax.patches:
            ax.annotate(f'{int(patch.get_height())}',
                        (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        fig = ax.get_figure()
        fig.tight_layout()

        plt.show()
        plt.close()

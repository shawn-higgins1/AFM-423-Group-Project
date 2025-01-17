�	�����7@�����7@!�����7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�����7@�!�uqs4@1��)t^c�?A��ڊ�e�?I���
X@*	�����LF@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!xI@<@)'�����?1��;��7@:Preprocessing2U
Iterator::Model::ParallelMapV2�g��s��?!"^N�7@)�g��s��?1"^N�7@:Preprocessing2F
Iterator::Model;�O��n�?!oAV�-D@)���_vO~?1�$ ���0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate��~j�t�?!g70�L5@)Ǻ���v?1����)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice�q����o?!�k�J!}!@)�q����o?1�k�J!}!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�]K�=�?!����M@)a��+ei?1m�J!}�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!RH_�T
@)ŏ1w-!_?1RH_�T
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!��p8@)Ǻ���V?1����	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���P�X@QM>ͫ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�!�uqs4@�!�uqs4@!�!�uqs4@      ��!       "	��)t^c�?��)t^c�?!��)t^c�?*      ��!       2	��ڊ�e�?��ڊ�e�?!��ڊ�e�?:	���
X@���
X@!���
X@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���P�X@yM>ͫ�?�"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���	�?!���	�?"5
sequential/dense/MatMulMatMul)�>�?!��*��?0"C
%gradient_tape/sequential/dense/MatMulMatMula��?!a��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul������?!�N�G�?0"7
sequential/dense_1/MatMulMatMulͩ1#��?!�k���?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�@��&?�?!�lo��?"7
sequential/dense_2/MatMulMatMul�I��?!4u�/Q�?0"!
Adam/PowPowa��?!����B:�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitcha��?!�F��K�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrada��?!�H�]�?Q      Y@Y>����/@aX�i��U@qPF��,�X@y�A�WF�?"�
both�Your program is POTENTIALLY input-bound because 86.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
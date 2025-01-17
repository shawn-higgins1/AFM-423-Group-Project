�	l��g��5@l��g��5@!l��g��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-l��g��5@��{b3@1b��!��?A	�c�?I���K�@*	�����YI@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenatea2U0*��?!��WV�B@)/n���?14{d[A@:Preprocessing2U
Iterator::Model::ParallelMapV2���S㥋?!1��k��:@)���S㥋?11��k��:@:Preprocessing2F
Iterator::Model8��d�`�?!w�d(�C@)-C��6z?1|1z�?)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-!?!�q`��-@)��_vOv?1��"AM%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip� �	��?!����_N@)Ǻ���f?1L�*:@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!4{d[@)/n��b?14{d[@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ZӼ�?!o_Y�KD@)a2U0*�S?1��WV�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!�wɃg�?)��H�}M?1�wɃg�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!L�*:�?)Ǻ���F?1L�*:�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIF_�X@Qs�\P���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��{b3@��{b3@!��{b3@      ��!       "	b��!��?b��!��?!b��!��?*      ��!       2		�c�?	�c�?!	�c�?:	���K�@���K�@!���K�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qF_�X@ys�\P���?�"5
sequential/dense/MatMulMatMul�-���ե?!�-���ե?0"C
%gradient_tape/sequential/dense/MatMulMatMul`a�h�?!vG�K��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�����B�?!�ų���?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�����B�?!)"��a`�?"7
sequential/dense_1/MatMulMatMul�����B�?!`a�h�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradF�s�i�?!��o����?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam`a�h�?!��Q�C�?"!
Adam/PowPow`a�h�?!�3�.��?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile�s���?!?�\~'��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam����?!`|�S�?Q      Y@Y{	�%��1@a�����T@q�'y$j�X@y������?"�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
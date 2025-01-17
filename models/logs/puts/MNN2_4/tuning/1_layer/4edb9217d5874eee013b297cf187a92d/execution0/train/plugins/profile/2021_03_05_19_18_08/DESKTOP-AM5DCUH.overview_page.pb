�	@�����5@@�����5@!@�����5@	B�1�?B�1�?!B�1�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6@�����5@�\��u3@1������?A�J�4�?I+j0@Y��:r�3�?*	����̌G@2U
Iterator::Model::ParallelMapV2��@��ǈ?!\�>;��9@)��@��ǈ?1\�>;��9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!���S:@)�g��s��?1�S
�[�6@:Preprocessing2F
Iterator::Model�j+��ݓ?!����/�D@)�<,Ԛ�}?1x��t�.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��_�L�?!|+�g�6@)F%u�{?1�^���,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!d���" @)ŏ1w-!o?1d���" @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��e�c]�?!Ja{+�gM@)-C��6j?1"��-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�O[h��@)��H�}]?1�O[h��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��@��ǈ?!\�>;��9@)_�Q�[?1�r���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9B�1�?I_��56�X@QP�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�\��u3@�\��u3@!�\��u3@      ��!       "	������?������?!������?*      ��!       2	�J�4�?�J�4�?!�J�4�?:	+j0@+j0@!+j0@B      ��!       J	��:r�3�?��:r�3�?!��:r�3�?R      ��!       Z	��:r�3�?��:r�3�?!��:r�3�?b      ��!       JGPUYB�1�?b q_��56�X@yP�����?�"5
sequential/dense/MatMulMatMul)Z���h�?!)Z���h�?0"C
%gradient_tape/sequential/dense/MatMulMatMulw_�χ�?!�\f�R��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�G�B�?!�.'��1�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�q)l��?!|�AL���?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�l��h�?!�[e,.�?"[
AArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_CastCast�l��h�?!��u~���?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�l��h�?!5���)�?"7
sequential/dense_1/MatMulMatMul�l��h�?!Ȃ��s��?0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�T�s�?!��,;��?"$
MaximumMaximum�<P���?!����)��?Q      Y@Y��/Ċ�0@a�	�N]�T@q���$�W@yAS��W�?"�
both�Your program is POTENTIALLY input-bound because 87.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 